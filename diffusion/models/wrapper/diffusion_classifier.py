"""Joint diffusion-and-classification training wrapper.

This wrapper expects the raw feature-routing and classifier head implemented by
``DiTClassifier`` and compatible convolutional/composed classifier networks.
It adds losses,
metrics, EMA use, timestep masking, and Keras train/test steps.

Runtime teachers supply frozen probability/noise targets; current classifier
and noise specialists can be separate from the previous student snapshot. Student
class heads can grow across continual tasks. Both mapped and online preparation preserve
an explicit optional replay-provenance tensor. Public fitting updates raw/EMA
weights and metric state; evaluate_ensemble_accuracy creates a fresh independent
metric for each evaluation call.
"""

import tensorflow as tf
from tensorflow.keras import callbacks, metrics, losses, models

import numpy as np

from math import ceil

import inspect

from typing import Callable, get_args, Literal, Sequence
from collections.abc import Mapping
from contextlib import nullcontext

from . import NetworkName, TrainType

from common.validation import require
from common.random import SeedStream
from common.runtime import derive_seed
from common.model import expand_classifier_head, validate_progressive_classifier_growth
from common.keras_compat import register_optimizer_variables

from autoencoder.variational_autoencoder import VariationalAutoencoder

from diffusion import TeacherName
from diffusion.models.wrapper.diffusion_model import DiffusionModel
from diffusion.metrics.ensemble_accuracy import EnsembleAccuracy


class DiffusionClassifier(DiffusionModel):
    """Train a compatible diffusion denoiser and image classifier jointly.

    The inherited diffusion objective updates the raw noise branch.  A second
    sparse-categorical objective trains the classifier probabilities, with
    optional example masking and optional classifier-side KL/class-token
    regularization.  ``DiTClassifier`` owns the architecture and all ``clf_*``
    layer attributes; this wrapper owns ``clf_*`` losses and metrics.

    Attributes:
        clf_loss_coef (tf.Tensor): Policy-variable-dtype scalar initialized from the
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
        use_clf_distil_loss (bool): True when a teacher and a positive
            ``clf_distil_loss_coef`` are present. Uses the primary classifier
            head when the student has no distillation token.
        use_clf_distil_ctr_loss (bool): Whether classifier regularizers use their
            frozen teacher target in ``"distil"`` or ``"both"`` mode.

    ``fit_teacher`` delegates native teachers to a cached
    DiffusionClassifier's fit (or fit_progressively), training both noise and
    classes with these constructor settings. Set trainable_teacher=True through
    kwargs to enable its separate compilation and optimizer. The inherited
    teacher_training option schedules native continual teacher fitting on each
    task or only the first task. Built, compiled ordinary Keras classifier teachers
    instead use their own optimizer, loss and metrics for explicit fit_teacher calls.
    """

    def __init__(
        self, 
        mask_by_nulls: bool | None = None, 
        mask_by_t_threshold: bool = False, 
        mask_t_percentage: int = 70, 
        clf_train_noisy_input_type: Literal["noisy", "clean"] = "noisy", 
        clf_train_class_input_type: Literal["null_class_only", "all_classes"] | None = None, 
        clf_train_batch_fraction: float = 0., 
        clf_train_noisified_max_timesteps: int | None = None, 
        clf_test_noisified_max_timesteps: int | None = None, 
        use_ensemble_loss_instead: bool = False, 
        clf_train_type: TrainType = "cond", 
        clf_loss_coef: float = 8.6e-3, 
        clf_distil_loss_coef: float = 0., 
        clf_acc_coef: float = .5, 
        ctr_acc_coef: float = 0., 
        clf_distil_acc_coef: float = .5, 
        clf_distil_type: Literal["hard", "soft"] = "hard", 
        clf_distil_temperature: float = 1., 
        clf_distil_scope: Literal[
            "old_classes", 
            "replay_only", 
            "current_and_replay"
        ] = "current_and_replay", 
        classifier_teacher_network: tf.keras.Model | None = None, 
        teacher_dynamic_classes: bool = False, 
        teacher_classifier_from_logits: bool = False, 
        previous_teacher_clf_loss_weight: float = 1., 
        current_teacher_clf_loss_weight: float = 1., 
        clf_distil_noisy_input_type: Literal["noisy", "clean"] = "noisy", 
        clf_distil_train_noisified_max_timesteps: int | None = None, 
        clf_distil_test_noisified_max_timesteps: int | None = None, 
        **kwargs: object
    ) -> None:
        """Initialize classifier-loss behavior around a raw classifier network.

        Creates classifier loss/accuracy trackers and resolves independent teacher
        attachments on top of DiffusionModel. Loss/metric accumulation uses the policy
        variable dtype; native logits/probabilities keep their producing layers' dtype.
        Ordinary teachers are Keras objects, not diffusion input tensors; the wrapper
        records their trainability and freezes them outside explicit teacher fitting.
        No classifier weights are trained by construction.

        Args:
            mask_by_nulls (bool | None): Select classifier loss/accuracy rows
                whose original diffusion CFG label is null after dropout.
                This mask is independent of classifier conditioning inputs.
                None preserves the historical True default; enabled masking
                requires ``p_uncond > 0``.
                Defaults to ``None``.
            mask_by_t_threshold (bool): Additionally select only examples with
                sampled ``t <= filter_t_threshold``.
                Defaults to ``False``.
            mask_t_percentage (int): Percentage in ``[0,100]`` used to construct the
                inclusive
                threshold ``ceil(percentage / 100 * timesteps) - 1``.  For
                T=1000 and 70, exactly timesteps 0 through 699 are selected;
                0 percent selects no examples.
                Defaults to ``70``.
            clf_train_noisy_input_type (Literal["noisy", "clean"]): ``"noisy"``
                selects the diffusion pass's ``x_t`` and timestep unless a
                classifier noising cap is supplied; ``"clean"``
                selects ``x0`` at timestep zero. The class-input selector
                independently chooses CFG or null labels. Noisy/all-classes
                with no cap reuses the primary forward outputs; other combinations
                performs one additional classifier pass in the same tape and
                optimizer update when ``clf_train_batch_fraction`` is zero.
                A positive fraction instead changes only allocated classifier
                rows within the single primary pass.
                Defaults to ``"noisy"``.
            clf_train_class_input_type (Literal["null_class_only", "all_classes"] | None):
                Classifier conditioning selector, never a row selector.
                ``all_classes`` uses the primary pass's post-dropout CFG labels;
                ``null_class_only`` uses null labels for every classifier input.
                Explicit values take precedence over ``clf_train_type`` and
                need no ``train_cfg_scale``. None maps legacy ``cond`` to
                ``all_classes`` and ``uncond`` to ``null_class_only``. Defaults
                to None, resolving to ``all_classes`` with the default train type.
            clf_train_batch_fraction (float): Finite fraction in ``[0,1]``.
                Zero preserves full-batch training and its optional additional
                classifier pass. A positive value randomly allocates
                ``max(1, floor(batch_size * fraction))`` rows to classification
                and the remaining rows to diffusion, using one raw-network pass.
                Input selectors change only classifier rows; original CFG/time
                masks further restrict their classifier losses. Each objective
                remains an independently normalized mean. Requires no training
                CFG scale or ensemble-loss replacement. Defaults to ``0.0``.
            clf_train_noisified_max_timesteps (int | None): Exclusive classifier
                noising cap, active in V1 only for ``"noisy"`` inputs. None
                preserves V1's diffusion inputs; zero selects clean timestep
                zero, -1 uses the full horizon, and positive caps sample
                ``[0, cap)`` independently of diffusion bounds. A supplied cap
                requires an additional classifier pass unless batch allocation
                is enabled. In V2, None retains clean-only training.
                Defaults to None.
            clf_test_noisified_max_timesteps (int | None): Equivalent evaluation
                cap. None preserves clean evaluation in both versions. V1
                ignores both caps for ``clf_train_noisy_input_type="clean"``.
                Defaults to None.
            use_ensemble_loss_instead (bool): Ignore the current forward pass's
                class probabilities for classifier loss and use
                a four-timestep raw-network ensemble on clean images instead.
                The resulting probabilities are also returned for accuracy.
                Defaults to ``False``.
            clf_train_type (TrainType): Legacy classifier conditioning choice
                when ``clf_train_class_input_type`` is omitted. ``cond`` maps
                to CFG labels; ``uncond`` maps to null labels and retains its
                historical ``train_cfg_scale`` requirement for noisy inputs
                when ``clf_train_batch_fraction`` is zero.
                An explicit class-input selector takes precedence. Defaults to
                ``'cond'``.
            clf_loss_coef (float): Scalar multiplier for classifier
                cross-entropy, default ``8.6e-3``.
                Defaults to ``0.0086``.
            clf_distil_loss_coef (float): Multiplier for classifier distillation.
                Uses the distillation-token head when present, otherwise the
                primary head. A positive value enables mapping with a teacher.
                Defaults to ``0.0``.
            clf_acc_coef (float): Primary-head coefficient used only for the
                wrapper's ``total_accuracy`` prediction.
                Defaults to ``0.5``.
            ctr_acc_coef (float): Classifier-regularizer coefficient used only
                for the wrapper's ``total_accuracy`` prediction.
                Defaults to ``0.0``.
            clf_distil_acc_coef (float): Distillation-head coefficient used only
                for the wrapper's ``total_accuracy`` prediction.
                Defaults to ``0.5``.
            clf_distil_type (Literal["hard", "soft"]): ``"hard"`` applies
                sparse cross-entropy to the teacher argmax; ``"soft"`` applies
                teacher-to-student KL divergence.
                Defaults to ``'hard'``.
            clf_distil_temperature (float): Positive soft-distillation
                temperature. ``1`` preserves the historical direct
                probability KL exactly; other values soften both teacher and
                student probabilities and apply the standard ``T**2`` scale.
                Hard distillation remains teacher-argmax cross-entropy.
                Defaults to ``1.0``.
            clf_distil_scope (Literal["old_classes", "replay_only", "current_and_replay"]):
                Teacher-target exposure. old_classes selects targets below the frozen
                teacher width; replay_only uses the explicit third dataset tensor;
                current_and_replay includes every classifier-selected row. Classifier
                timestep/null masks still intersect these scopes. In dual-teacher mode
                these historical scopes apply only to the previous teacher; the current
                teacher follows dual_teacher_scope and classifier row masks.
                Defaults to ``'current_and_replay'``.
            classifier_teacher_network (tf.keras.Model | None): Independent current-task
                classifier specialist, such as a fine-tuned EfficientNetV2L. It is
                runtime-only, frozen outside fit_teacher(teacher_name="classifier"),
                and cannot be combined with current_teacher_network. Defaults to None.
            teacher_dynamic_classes (bool): Grow an ordinary Keras teacher's final Dense
                head as fit_teacher discovers sparse dataset IDs. Existing columns,
                fine-tuning masks and optimizer state survive growth. Defaults to False.
            teacher_classifier_from_logits (bool): Interpret an image-only callable
                teacher's scores as logits and convert them to probabilities. False
                requires normalized probabilities. Native predict_class teachers
                already return probabilities and ignore this option. Defaults to False.
            previous_teacher_clf_loss_weight (float): Multiplier for previous-task
                classifier KD, before the common clf_distil_loss_coef. Defaults to 1.
                Defaults to ``1.0``.
            current_teacher_clf_loss_weight (float): Multiplier for current-task
                classifier KD. Zero disables this teacher's classifier objective.
                Defaults to 1.
                Defaults to ``1.0``.
            clf_distil_noisy_input_type (Literal["noisy", "clean"]): ``"noisy"``
                preserves the student's selected classifier images and timesteps for
                native teachers unless a teacher noising cap is supplied;
                ``"clean"`` ignores teacher caps and uses clean images at timestep zero,
                independently of student noising. Applies to classifier targets in
                V1 and V2, including evaluation and all teacher roles. Ordinary
                image-only teachers always receive clean images. Noise-distillation
                inputs and class conditioning are unchanged. Defaults to ``"noisy"``.
            clf_distil_train_noisified_max_timesteps (int | None): Exclusive native
                classifier-teacher noising cap during student training. None preserves
                the student's selected classifier inputs; zero selects exact clean
                images, -1 uses the full horizon, and positive caps sample [0, cap)
                from clean images independently of student bounds. Active only with
                clf_distil_noisy_input_type="noisy". All native teacher roles share
                one draw; ordinary image-only teachers stay clean. Defaults to None.
            clf_distil_test_noisified_max_timesteps (int | None): Equivalent native
                classifier-teacher cap during student evaluation. None preserves
                the student's selected evaluation inputs. Ignored with
                clf_distil_noisy_input_type="clean". Defaults to None.
            **kwargs (object): Arguments forwarded to ``DiffusionModel``.  Required in
                normal use is ``network=DiTClassifier(...)``; supported wrapper
                keys include EMA/scheduler/CFG settings, all four diffusion loss
                coefficients and train types, timestep bounds, resize options,
                ``swap_noise_image``, ``seed``, and standard Keras ``Model`` keys
                ``name``, ``trainable``, ``dtype``, and ``dynamic``.

        Returns:
            None: Classifier coefficients, threshold, and loss flags are
            initialized alongside metric trackers before :meth:`compile`.

        Raises:
            AssertionError: Classifier masks, coefficients, temperature, train type, or
                teacher requirements are invalid.
            ValueError: The runtime teacher is incompatible with the student protocol.
        """

        object.__setattr__(self, "classifier_teacher_network", classifier_teacher_network)
        object.__setattr__(self, "teacher_dynamic_classes", teacher_dynamic_classes)
        super().__init__(**kwargs)
        self._save_init_args(
            locals(), 
            exclude=(
                "self", "kwargs", "__class__", 
                "classifier_teacher_network"
            )
        )
        self.mask_by_nulls = True if self.mask_by_nulls is None else self.mask_by_nulls
        self._init_config["mask_by_nulls"] = self.mask_by_nulls
        self._check_clf_assertions({**locals(), "mask_by_nulls": self.mask_by_nulls})
        DiffusionClassifier._create_metrics(self)

        # Infer omitted classifier conditioning from the legacy train-type API.
        if self.clf_train_class_input_type is None:
            # Retain validation for legacy noisy unconditional callers only.
            if self.clf_train_type == "uncond" and \
            self.clf_train_noisy_input_type == "noisy" \
            and self.clf_train_batch_fraction == 0.:
                require(
                    self.use_cfg and self.train_cfg_scale is not None, 
                    "Unconditional classifier training requires CFG and train_cfg_scale."
                )
            self.clf_train_class_input_type = (
                "null_class_only" if self.clf_train_type == "uncond" 
                else "all_classes"
            )

        # Opt-in allocation owns an independent checkpointed stream; legacy weights stay unchanged.
        if self.clf_train_batch_fraction > 0.:
            self._random_streams["classifier_batch"] = SeedStream(
                derive_seed(self.seed, "diffusion", "classifier_batch"), 
                name=f"{self.name}__classifier_batch_random"
            )

        self.clf_train_noisified_max_timesteps = 0 if self.clf_train_noisified_max_timesteps is None \
                                                else int(self.clf_train_noisified_max_timesteps)
        self.clf_train_noisified_max_timesteps = self.timesteps if self.clf_train_noisified_max_timesteps == -1 \
                                                else self.clf_train_noisified_max_timesteps
        self.clf_test_noisified_max_timesteps = 0 if self.clf_test_noisified_max_timesteps is None \
                                                else int(self.clf_test_noisified_max_timesteps)
        self.clf_test_noisified_max_timesteps = self.timesteps if self.clf_test_noisified_max_timesteps == -1 \
                                                else self.clf_test_noisified_max_timesteps
        for name in (
            "clf_distil_train_noisified_max_timesteps", 
            "clf_distil_test_noisified_max_timesteps"
        ):
            value = getattr(self, name)
            value = None if value is None else int(value)
            setattr(self, name, self.timesteps if value == -1 else value)
        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        self.clf_loss_coef = tf.constant(
            self.clf_loss_coef, 
            dtype=stable_dtype
        )
        self.clf_distil_loss_coef = tf.constant(
            self.clf_distil_loss_coef, 
            dtype=stable_dtype
        )
        self.filter_t_threshold = tf.constant(
            ceil(self.mask_t_percentage / 100 * self.timesteps) - 1, 
            dtype=tf.int32
        )

        self._init_config.pop("teacher_network", None)
        self._init_config.pop("current_teacher_network", None)
        self._init_config.pop("classifier_teacher_network", None)
        self._init_config.pop("noise_teacher_network", None)
        self.set_teacher_network(self.teacher_network)
        for teacher_name in ("current", "classifier", "noise"):
            self.set_current_teacher_network(
                self.get_teacher_network(teacher_name), 
                class_ids=getattr(self, teacher_name + "_teacher_class_ids"), 
                task_class_ids=getattr(self, teacher_name + "_teacher_task_class_ids"), 
                teacher_name=teacher_name
            )

    def _check_clf_assertions(self, local_vars: dict[str, object]) -> None:
        """Validate classifier training choices after base initialization.

        Null-label masking requires classifier-free guidance dropout, independent
        of the classifier's input conditioning. Explicit clean/null input choices
        own one additional prediction and cannot be replaced by an ensemble.

        Args:
            local_vars (dict[str, object]): Classifier constructor arguments.

        Returns:
            None: Compatible classifier constructor settings pass without mutation.

        Raises:
            AssertionError: Mask/dropout combinations, coefficients, temperature, KD scope,
                train type, teacher presence, or ensemble horizon are invalid.
        """

        # Null-only masking requires a nonzero probability of null labels.
        if local_vars["mask_by_nulls"]:
            require(
                self.p_uncond > 0., 
                "null_class_only / mask_by_nulls requires p_uncond > 0."
            )

        require(
            0 <= local_vars["mask_t_percentage"] <= 100, 
            "mask_t_percentage must be in [0, 100]."
        )

        for name in (
            "clf_train_noisified_max_timesteps", 
            "clf_test_noisified_max_timesteps", 
            "clf_distil_train_noisified_max_timesteps", 
            "clf_distil_test_noisified_max_timesteps"
        ):
            value = local_vars[name]
            value = None if value is None else int(value)
            require(
                value is None or -1 <= value <= self.timesteps, 
                f"{name} must be None or in [-1, timesteps]."
            )

        require(
            local_vars["clf_train_class_input_type"] in (
                None, 
                "null_class_only", 
                "all_classes"
            ), 
            "clf_train_class_input_type must be 'null_class_only' or 'all_classes'."
        )

        require(
            isinstance(local_vars["clf_train_batch_fraction"], (int, float, np.integer, np.floating))
            and not isinstance(local_vars["clf_train_batch_fraction"], (bool, np.bool_))
            and np.isfinite(local_vars["clf_train_batch_fraction"])
            and 0. <= local_vars["clf_train_batch_fraction"] <= 1., 
            "clf_train_batch_fraction must be a finite number in [0, 1]."
        )

        for name in (
            "clf_loss_coef", "clf_distil_loss_coef", 
            "clf_acc_coef", "ctr_acc_coef", 
            "clf_distil_acc_coef", 
            "previous_teacher_clf_loss_weight", 
            "current_teacher_clf_loss_weight"
        ):
            value = local_vars[name]
            require(
                np.isfinite(value) and value >= 0., 
                f"{name} must be finite and nonnegative."
            )

        require(
            isinstance(local_vars["teacher_classifier_from_logits"], bool), 
            "teacher_classifier_from_logits must be a bool."
        )
        require(
            local_vars["clf_distil_type"] in ("hard", "soft"), 
            "clf_distil_type must be either 'hard' or 'soft'."
        )
        require(
            local_vars["clf_distil_noisy_input_type"] in ("noisy", "clean"), 
            "clf_distil_noisy_input_type must be 'noisy' or 'clean'."
        )
        require(
            np.isfinite(local_vars["clf_distil_temperature"]) and
            local_vars["clf_distil_temperature"] > 0., 
            "clf_distil_temperature must be finite and positive."
        )
        require(
            local_vars["clf_distil_scope"] in (
                "old_classes", "replay_only", 
                "current_and_replay"
            ), 
            "clf_distil_scope must be 'old_classes', "
            "'replay_only', or 'current_and_replay'."
        )

        # Only enabled teacher roles require targets for a positive classifier objective.
        if (local_vars["previous_teacher_clf_loss_weight"] > 0. or 
        local_vars["current_teacher_clf_loss_weight"] > 0.) and (
        local_vars["clf_distil_loss_coef"] > 0. or (
        local_vars["kwargs"].get("ctr_loss_coef", 0.) > 0. and(
        getattr(self.network, "clf_cls_token_regularizer_kwargs", None
        ) or getattr(self.network, "cls_token_regularizer_kwargs", {})
        ).get("train_type", "normal") in ("distil", "both")
        )):
            require(
                self.teacher_network is not None
                or self.current_teacher_network is not None
                or self.classifier_teacher_network is not None or self.defer_teacher, 
                "teacher_network is required for distillation "
                "training unless defer_teacher=True."
            )

        # The four-step ensemble requires at least four available timesteps.
        if local_vars["use_ensemble_loss_instead"]:
            require(
                self.timesteps >= 4, 
                "use_ensemble_loss_instead requires at least four timesteps."
            )

        require(
            local_vars["clf_train_type"] in get_args(TrainType), 
            f"clf_train_type can only be one of {TrainType}."
        )

        require(
            local_vars["clf_train_noisy_input_type"] in ("noisy", "clean"), 
            "clf_train_noisy_input_type must be 'noisy' or 'clean'."
        )

        # Null conditioning requires an unconditional label in the network vocabulary.
        if local_vars["clf_train_class_input_type"] == "null_class_only":
            require(self.use_cfg, "Null classifier inputs require CFG.")

        self._check_classifier_input_policy(local_vars)

        # Partitioned batches require a single ordinary network prediction.
        if local_vars["clf_train_batch_fraction"] > 0.:
            require(
                self.train_cfg_scale is None, 
                "Positive clf_train_batch_fraction requires train_cfg_scale=None."
            )
            require(
                not local_vars["use_ensemble_loss_instead"], 
                "Positive clf_train_batch_fraction cannot use ensemble-loss replacement."
            )

    def _check_classifier_input_policy(self, local_vars: dict[str, object]) -> None:
        """Keep explicit V1 classifier input policies from being bypassed by an ensemble.

        Args:
            local_vars (dict[str, object]): Constructor options for classifier
                corruption caps, clean/null input selection, and ensemble loss.

        Returns:
            result (None): Leaves configuration unchanged when the selected paths agree.

        Raises:
            AssertionError: Ensemble replacement would ignore explicit classifier inputs.
        """

        # Ensembles perform their own noising and cannot honor selected images or caps.
        if local_vars["clf_train_noisy_input_type"] == "clean" \
        or local_vars["clf_train_class_input_type"] == "null_class_only" \
        or local_vars["clf_train_noisified_max_timesteps"] is not None \
        or local_vars["clf_test_noisified_max_timesteps"] is not None:
            require(
                not local_vars["use_ensemble_loss_instead"], 
                "Explicit clean/null classifier inputs or noising caps "
                "cannot replace their prediction with an ensemble."
            )

    def _refresh_loss_flags(self) -> None:
        """Refresh diffusion and classifier auxiliary-loss availability.

        The base flags describe the diffusion branch. This override adds the
        classifier KL and class-token regularizer flags from the classifier's
        own reshaper and regularizer metadata. It is called at construction and
        again after progressive depth growth.

        Args:
            None.

        Returns:
            None: Classifier and distillation loss flags are updated.

        Raises:
            ValueError: Enabled distillation or token supervision requires same-pass logits but the native
                call/predict_class signatures do not expose return_logits. No network forward pass
                is run.
        """

        super()._refresh_loss_flags()

        # Expose classifier KL availability only when the network declares classifier reshaper
        # metadata.
        self.use_clf_kl_loss = bool(
            self.kl_loss_coef > 0. and 
            self.network.clf_reshaper_kwargs.get("add_kl", False)
        ) if getattr(self.network, "clf_reshaper_kwargs", None) is not None else None
        # Expose token-loss availability only when the network declares classifier regularizer
        # depths.
        self.use_clf_ctr_loss = bool(
            self.ctr_loss_coef > 0. and 
            len(self.network.clf_cls_token_regularizer_ids) > 0
        ) if getattr(self.network, "clf_cls_token_regularizer_ids", None) is not None else None
        self.use_clf_distil_loss = bool(
            bool(self._classifier_teacher_specs()) and
            self.clf_distil_loss_coef > 0.
        )

        regularizer_kwargs = getattr(
            self.network, 
            "clf_cls_token_regularizer_kwargs", 
            None
        )
        regularizer_kwargs = getattr(
            self.network, 
            "cls_token_regularizer_kwargs", 
            {}
        ) if regularizer_kwargs is None else regularizer_kwargs
        self.use_clf_distil_ctr_loss = bool(
            bool(self._classifier_teacher_specs()) and
            self.ctr_loss_coef > 0. and
            regularizer_kwargs.get("train_type", "normal") in (
                "distil", "both"
            )
        )

        self.use_classifier_distil = (
            self.use_clf_distil_loss or 
            self.use_clf_distil_ctr_loss
        )
        self.use_total_accuracy = bool(
            self.use_clf_distil_loss and 
            self.clf_distil_acc_coef > 0.
        ) or bool(
            self.use_clf_ctr_loss and 
            self.ctr_acc_coef > 0.
        )

        # Cross-entropy and KL need scores from the same stochastic student pass.
        if self.use_classifier_distil or self.use_clf_ctr_loss:
            self._require_student_classifier_logits()
            self.use_logits_instead = {"return_logits": True}
        # Ordinary primary-head classification keeps its probability API.
        else:
            self.use_logits_instead = {}

    def _require_student_classifier_logits(self) -> None:
        """Require both native classifier paths to expose same-pass logits.

        Inspects signatures without running the network or creating variables. Merely
        accepting arbitrary keywords is insufficient: return_logits must be explicit
        in both call and predict_class.

        Args:
            None.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: Either callable lacks an explicit return_logits parameter. TypeError or
                ValueError from inspecting an unsupported callable signature is propagated.
        """

        for method in (self.network.call, self.network.predict_class):
            parameters = inspect.signature(method).parameters

            # Stable losses require exact scores from the stochastic student pass.
            if "return_logits" not in parameters:
                raise ValueError(
                    "Classifier KD and token CE require same-pass return_logits support."
                )

    def _classifier_teacher_specs(self) -> tuple[dict[str, object], ...]:
        """Describe active frozen class teachers in previous/current role order.

        Keeps only attached roles with a positive classifier loss weight. Dynamic Keras
        head columns are mapped through persistent dataset IDs to student IDs, with -1
        for untaught/absent classes. Native previous snapshots retain leading-column
        semantics. Reading these specifications does not freeze, call or mutate a model.

        Args:
            None.

        Returns:
            tuple[dict[str, object], ...]: Each mapping contains role (str), network (Keras object),
                weight (float), class_ids (tuple[int, ...] | None) and task_class_ids (tuple[int,
                ...] | None). Empty when no classifier role is active.

        Raises:
            This metadata resolver raises no explicit exceptions. Invalid dynamically supplied
                output/vocabulary metadata may fail while reading a head width or indexing its
                columns.
        """

        specs = []
        # Persistent image teachers identify columns by dataset class, not student order.
        if self.teacher_network is not None and self.previous_teacher_clf_loss_weight > 0.:
            taught = getattr(self.teacher_network, "_diffusion_task_class_ids", None)
            mapping = getattr(self.teacher_network, "_diffusion_seen_classes", {})
            class_ids = None
            # Untaught capacity and classes absent from the student have no shared support.
            if getattr(self.teacher_network, "_diffusion_dynamic_classes", False):
                class_ids = [-1] * int(self.teacher_network.output_shape[-1])
                for label, column in mapping.items():
                    class_ids[column] = self.seen_classes.get(label, -1) \
                                        if self.network.dynamic_num_classes else int(label)

                class_ids = tuple(class_ids)
                taught = tuple(class_id for class_id in class_ids if class_id >= 0)
            # Native snapshots retain their shared leading vocabulary.
            elif taught is None:
                taught = tuple(mapping.values()) if mapping else None
            
            specs.append(dict(
                role="previous", 
                network=self.teacher_network, 
                weight=self.previous_teacher_clf_loss_weight, 
                class_ids=class_ids, 
                task_class_ids=taught
            ))

        current = self._current_teacher_spec("classifier")
        # A shared current teacher or classifier specialist owns its own class mapping.
        if current is not None and self.current_teacher_clf_loss_weight > 0.:
            specs.append(current)

        return tuple(specs)

    def _uses_mapped_classifier_teachers(self) -> bool:
        """Choose the structured target contract for independent class vocabularies.

        Args:
            None.

        Returns:
            bool: True if a current classifier teacher is attached (even with zero weight) or the
                previous ordinary teacher advertises dynamic columns; False for the legacy
                single-prefix target contract. No state changes.

        Raises:
            This predicate raises no explicit exceptions.
        """

        return self._current_teacher_spec("classifier") is not None or bool(getattr(
            self.teacher_network, 
            "_diffusion_dynamic_classes", 
            False
        ))

    def _predict_teacher_labels(
        self, 
        x_t: tf.Tensor, 
        t: tf.Tensor, 
        labels: tf.Tensor, 
        clean_images: tf.Tensor | None = None, 
        training: bool = True
    ) -> tf.Tensor | tuple[tf.Tensor, ...]:
        """Prepare independent frozen class targets from the selected teacher objects.

        An ordinary teacher may build lazy layers on first inference and is frozen
        again afterwards. Target creation does not apply teacher/student gradients.

        Args:
            x_t (tf.Tensor): Floating classifier images ``[B,H,W,C]`` for native
                timestep-aware teachers.
            t (tf.Tensor): Integer timestep IDs ``[B]`` aligned with x_t.
            labels (tf.Tensor): Integer student condition IDs ``[B]``; current
                teacher conditions are remapped through its local class vocabulary.
            clean_images (tf.Tensor | None): Original floating images ``[B,H,W,C]``;
                required by ordinary image-only classifiers and native teachers when
                clf_distil_noisy_input_type="clean" or a teacher noising cap is active.
                Images use model coordinates. None is valid only for native teachers
                using uncapped "noisy" inputs. Defaults to ``None``.
            training (bool): Select the teacher training cap when True, evaluation
                cap otherwise. Teachers always run frozen inference. Defaults to True.

        Returns:
            targets (tf.Tensor | tuple[tf.Tensor, ...]): Detached probabilities
                ``[B,teacher_width]`` per role. A lone previous teacher retains the
                tensor API unless its Keras head has a dynamic class mapping.
                Dynamic or dual teachers select a role-ordered tuple.
                Native dtypes are retained; half-precision callable scores use float32.

        Raises:
            ValueError: A teacher needs clean_images, returns unsupported score
                structure/dtype, or its role mapping is invalid.
            tf.errors.InvalidArgumentError: A returned probability batch has invalid rank, width,
                batch size, finite values or normalization. Native teacher execution errors
                propagate.
        """

        max_timesteps = self.clf_distil_train_noisified_max_timesteps if training \
                        else self.clf_distil_test_noisified_max_timesteps
        # Draw one independent corruption for all active native classifier teachers.
        if self.clf_distil_noisy_input_type == "noisy" and max_timesteps is not None \
        and any(callable(getattr(spec["network"], "predict_class", None))
                for spec in self._classifier_teacher_specs()):
            # Teacher corruption must start from clean images, never the student's x_t.
            if clean_images is None:
                raise ValueError("Capped classifier distillation requires clean_images (x0).")
            x_t, _, t = self.noisify(
                clean_images, min_timesteps=0, max_timesteps=max_timesteps
            )

        # Teachers with shared positional columns retain the legacy tensor target API.
        if not self._uses_mapped_classifier_teachers():
            return self._predict_single_teacher_labels(x_t, t, labels, clean_images)

        predictions = []
        for spec in self._classifier_teacher_specs():
            teacher = spec["network"]
            teacher_conditions = self._remap_teacher_conditions(labels, spec["class_ids"])
            limit = getattr(teacher, "num_labels", None)

            # Unknown conditional IDs use the native null condition before embedding lookup.
            if limit is not None:
                teacher_conditions = tf.where(
                    teacher_conditions < tf.cast(limit, teacher_conditions.dtype), 
                    teacher_conditions, tf.zeros_like(teacher_conditions)
                )

            predictions.append(self._predict_single_teacher_labels(
                x_t, 
                t, 
                teacher_conditions, 
                clean_images, 
                teacher_network=teacher
            ))

        return tuple(predictions)

    def _classifier_teacher_support(
        self, 
        probabilities: tf.Tensor, 
        spec: dict[str, object], 
        student_width: tf.Tensor | int
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Restrict teacher columns while retaining fixed class-axis widths for XLA.

        Args:
            probabilities (tf.Tensor): Floating teacher probabilities ``[B,Ct]``.
            spec (dict[str, object]): Role metadata with optional class_ids column
                mapping and task_class_ids taught subset in student class IDs.
            student_width (tf.Tensor | int): Scalar current student class count.

        Returns:
            support (tuple[tf.Tensor, tf.Tensor]): Renormalized same-dtype teacher
                probabilities ``[B,K]`` and int32 student class IDs ``[K]`` after
                excluding absent/untaught columns. Zero retained mass remains zero
                for the caller's eligibility-aware mass check.

        Raises:
            tf.errors.InvalidArgumentError: Class-map width differs from teacher
                output width or no shared class support remains.
        """

        ids = tf.range(
            tf.shape(probabilities)[1], 
            dtype=tf.int32
        ) if spec["class_ids"] is None else tf.constant(spec["class_ids"], tf.int32)
        tf.debugging.assert_equal(
            tf.size(ids), 
            tf.shape(probabilities)[1], 
            message="Teacher class_ids must match its output width."
        )

        teacher_width = probabilities.shape[-1] if probabilities.shape.rank is not None else None
        static_student_width = tf.get_static_value(student_width)

        # Declared head widths permit constant gathers, avoiding XLA's dynamic mask bounds.
        if teacher_width is not None and static_student_width is not None:
            source_ids = tuple(range(int(teacher_width))) if spec["class_ids"] is None \
                        else tuple(spec["class_ids"])
            taught = None if spec["task_class_ids"] is None else set(spec["task_class_ids"])
            columns = tuple(index for index, class_id in enumerate(source_ids)
                    if 0 <= class_id < int(static_student_width)
                    and (taught is None or class_id in taught))
            ids = tf.constant(tuple(source_ids[index] for index in columns), tf.int32)
            selected = tf.gather(probabilities, tf.constant(columns, tf.int32), axis=1)
        # Unknown callable head widths retain the ordinary graph-mode dynamic vocabulary path.
        else:
            valid = tf.logical_and(ids >= 0, ids < tf.cast(student_width, tf.int32))
            # Warm-start teachers retain a full head, but only newly taught columns are targets.
            if spec["task_class_ids"] is not None:
                taught_ids = tf.constant(spec["task_class_ids"], tf.int32)
                valid = tf.logical_and(valid, tf.reduce_any(ids[:, None] == taught_ids[None, :], axis=1))

            ids = tf.boolean_mask(ids, valid)
            selected = tf.boolean_mask(probabilities, valid, axis=1)

        tf.debugging.assert_positive(
            tf.size(ids), 
            message="Teacher and student need shared class support."
        )
        selected = tf.math.divide_no_nan(
            selected, 
            tf.reduce_sum(selected, axis=1, keepdims=True)
        )

        return selected, ids

    def _classifier_teacher_mask(
        self, 
        classes: tf.Tensor | None, 
        replay_mask: tf.Tensor | None, 
        support: tf.Tensor, 
        batch_size: tf.Tensor, 
        scope: str, 
        role: str = "previous"
    ) -> tf.Tensor:
        """Apply past-teacher scopes without suppressing independently trained current targets.

        Args:
            classes (tf.Tensor | None): Integer undropped student class IDs ``[B]``;
                needed for task or old-class selection.
            replay_mask (tf.Tensor | None): Boolean/numeric provenance ``[B]``;
                required by previous-role replay_only selection.
            support (tf.Tensor): Int32 taught student IDs ``[K]``.
            batch_size (tf.Tensor): Integer scalar B used to initialize all-row support.
            scope (str): Previous-role old_classes, replay_only, or current_and_replay
                policy. Current-role targets override it to current_and_replay.
            role (str): Previous or current teacher role; defaults to previous.

        Returns:
            eligible (tf.Tensor): Boolean row selector ``[B]`` after task membership
                and requested replay provenance are intersected.

        Raises:
            ValueError: Required class targets or replay provenance are absent.
        """

        # Historical old/replay scopes belong to the previous-task retention objective.
        scope = scope if role == "previous" else "current_and_replay"
        mask = tf.ones(tuple([batch_size]), tf.bool)

        # Task-local supervision and explicit old_classes scopes use the actual mapped support.
        if (self._current_teacher_spec("classifier") is not None and self.dual_teacher_scope == "task") \
                or scope == "old_classes":
            # Class membership is undefined without the original undropped targets.
            if classes is None:
                raise ValueError("Dual task-scoped distillation requires classes.")
            
            mask = tf.reduce_any(
                tf.cast(tf.reshape(classes, (-1, 1)), tf.int32) == support[None, :], 
                axis=1
            )

        # Replay scopes retain their explicit provenance contract.
        if scope == "replay_only":
            # Class labels alone cannot establish whether an example came from replay.
            if replay_mask is None:
                raise ValueError(
                    "replay_mask is required for replay_only distillation."
                )
            
            mask = tf.logical_and(
                mask, 
                tf.cast(tf.reshape(replay_mask, tuple([-1])), tf.bool)
            )
        
        return mask

    def _compute_dual_classifier_distillation(
        self, 
        targets: tuple[tf.Tensor, ...], 
        student: tf.Tensor, 
        loss_type: str, 
        temperature: float, 
        scope: str, 
        mask: tf.Tensor | None, 
        classes: tf.Tensor | None, 
        replay_mask: tf.Tensor | None, 
        student_logits: tf.Tensor | None, 
        allocator: Callable | None, 
        x0: tf.Tensor | None, 
        update_metrics: bool, 
        teacher_loss_weight: float | None = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Sum independent mapped CE/KL terms; never average targets before hard KD.

        Args:
            targets (tuple[tf.Tensor, ...]): Detached teacher probabilities
                ``[B,Ct]`` in active previous/current role order.
            student (tf.Tensor): Floating student probabilities ``[B,Cs]``.
            loss_type (str): hard chooses teacher argmax CE; soft chooses KL.
            temperature (float): Positive soft-KD temperature with squared scaling.
            scope (str): Previous-role old_classes/replay_only/current_and_replay policy.
            mask (tf.Tensor | None): Optional fractional classifier row weights ``[B]``.
            classes (tf.Tensor | None): Integer student targets ``[B]`` for eligibility.
            replay_mask (tf.Tensor | None): Explicit Boolean/numeric replay rows ``[B]``.
            student_logits (tf.Tensor | None): Same-pass scores ``[B,Cs]``; None
                permits probability-derived logits only when probabilities are positive.
            allocator (Callable | None): Optional independent-head row-loss reducer;
                None uses the standard selected-row mean.
            x0 (tf.Tensor | None): Clean images ``[B,H,W,C]`` forwarded to allocator.
            update_metrics (bool): Record each role's unweighted loss and population.
            teacher_loss_weight (float | None): Optional coefficient override for a
                single mapped teacher. Dual roles retain their configured weights.
                None (default) retains each configured role coefficient; a numeric override applies only
                when there is exactly one mapped role.
                Defaults to ``None``.

        Returns:
            result (tuple[tf.Tensor, tf.Tensor]): Policy-variable-dtype scalar
                weighted sum of role losses and the unchanged student probabilities.
                Each role keeps its own support, target, mask, and normalization.

        Raises:
            ValueError: Target count differs from active roles or required scope
                metadata is missing.
            tf.errors.InvalidArgumentError: An eligible teacher row has zero mass
                on taught/shared columns, or teacher/student shapes disagree.
        """

        specs = self._classifier_teacher_specs()
        # Reject stale mapped batches after a role was attached or removed.
        if len(targets) != len(specs):
            raise ValueError(
                "Mapped classifier targets do not match the attached teacher roles."
            )

        total = tf.zeros((), dtype=self.dtype_policy.variable_dtype)
        static_width = student.shape[-1] if student.shape.rank is not None else None
        width = int(static_width) if static_width is not None else tf.shape(student)[1]
        for probabilities, spec in zip(targets, specs):
            probabilities, support = self._classifier_teacher_support(probabilities, spec, width)
            teacher_mask = self._classifier_teacher_mask(
                classes, 
                replay_mask, 
                support, 
                tf.shape(student)[0], 
                scope, 
                role=spec["role"]
            )
            weights = tf.cast(teacher_mask, self.dtype_policy.variable_dtype)
            # Preserve fractional classifier masks in each independently normalized objective.
            if mask is not None:
                weights *= tf.cast(mask, weights.dtype)

            static_support = tf.get_static_value(support)
            # A constant full-width permutation keeps target padding and its gradient XLA-safe.
            if static_width is not None and static_support is not None:
                support_ids = tuple(int(class_id) for class_id in static_support)
                remaining = tuple(class_id for class_id in range(int(static_width))
                                  if class_id not in support_ids)
                order = tf.constant(support_ids + remaining, tf.int32)
            # Runtime-width heads continue to use the existing ordinary graph permutation.
            else:
                remaining = tf.boolean_mask(tf.range(width), tf.logical_not(
                            tf.reduce_any(tf.range(width)[:, None] == support[None, :], axis=1)))
                order = tf.concat((support, remaining), axis=0)

            # Reordering all student columns preserves its full softmax denominator.
            loss, _ = self._compute_single_classifier_distillation(
                probabilities, 
                tf.gather(student, order, axis=1), 
                clf_distil_type=loss_type, 
                clf_distil_temperature=temperature, 
                clf_distil_scope="current_and_replay", 
                clf_distil_loss_mask=weights, 
                classes=classes, 
                replay_mask=replay_mask, 
                kd_loss_allocator=allocator, 
                x0=x0, 
                student_logits=tf.gather(student_logits, order, axis=1)
                            if student_logits is not None else None, 
                teacher_loss_weight=1.
            )
            weight = teacher_loss_weight if len(specs) == 1 and teacher_loss_weight is not None \
                    else spec["weight"]
            total += tf.cast(tf.convert_to_tensor(weight, dtype_hint=total.dtype), total.dtype) * loss

            # Separate population-weighted trackers make the sum invariant to batch partitioning.
            if update_metrics:
                tracker = getattr(self, spec["role"] + "_teacher_clf_loss_tracker")
                tracker.update_state(loss, sample_weight=tf.reduce_sum(weights))

        return total, student

    def _predict_single_teacher_labels(
        self, 
        x_t: tf.Tensor, 
        t: tf.Tensor, 
        labels: tf.Tensor, 
        clean_images: tf.Tensor | None = None, 
        teacher_network: tf.keras.Model | None = None
    ) -> tf.Tensor:
        """Return frozen teacher probabilities for one prepared batch.

        clean_images is in wrapper model coordinates, not raw teacher pixels. For an
        ordinary classifier it is restored with postprocess (without clipping), then
        passed in inference mode. Callable float16/bfloat16 scores are promoted to
        float32 before logits/probability validation; other floating/native probability
        dtypes are preserved. All returned scores are stop-gradient. A lazily built
        ordinary teacher is frozen again after its call.

        Args:
            x_t (tf.Tensor): Clean or noisified teacher images.
            t (tf.Tensor): Per-example timestep IDs.
            labels (tf.Tensor): Condition IDs supplied to a native teacher.
            clean_images (tf.Tensor | None): Clean images in wrapper model coordinates, required
                for an image-only callable teacher or clf_distil_noisy_input_type="clean".
                Native clean targets use timestep zero; "noisy" preserves x_t and t.
                None (default) is ignored only for native teachers using "noisy" inputs.
                Defaults to ``None``.
            teacher_network (tf.keras.Model | None): Explicit independent teacher;
                None selects the attached previous-task teacher.
                Defaults to ``None``.

        Returns:
            tf.Tensor: Stopped-gradient probabilities [B, teacher_class_count].
            Native prediction dtypes are retained; float16/bfloat16 image-only
            scores are promoted to float32 for normalization. The frozen
            vocabulary may be narrower or wider than the student's class count.

        Raises:
            ValueError: Required clean_images is absent, or an ordinary teacher's result is not
                one dense floating class-score tensor.
            tf.errors.InvalidArgumentError: Score rank/batch/width or finite/probability checks
                fail. A native predict_class failure propagates unchanged.
            ValueError: A statically known score rank is incompatible with the required rank-two
                output; tf.debugging.assert_rank can fail during tracing before a runtime assertion
                is created.
        """

        teacher = self.teacher_network if teacher_network is None else teacher_network
        # Native classifiers keep their timestep and condition-aware contract.
        if callable(getattr(teacher, "predict_class", None)):
            # Clean classifier targets are independent of the student's corruption.
            if self.clf_distil_noisy_input_type == "clean":
                # Missing x0 must not silently fall back to the student's noisy images.
                if clean_images is None:
                    raise ValueError("Clean classifier distillation requires clean_images (x0).")
                x_t, t = clean_images, tf.zeros_like(t)
            with self._teacher_inference_scope(teacher):
                teacher_labels = teacher.predict_class(
                    (x_t, t, labels), 
                    max_encoder_num=None, 
                    training=False
                )
        # An ordinary classifier always sees the original clean image batch.
        else:
            # Never silently substitute a noisy batch when a caller omits x0.
            if clean_images is None:
                raise ValueError("An image-only teacher requires clean_images (x0).")

            with self._teacher_inference_scope(teacher):
                # Ordinary classifiers receive the same external pixels used by fit_teacher.
                teacher_images = self.postprocess(clean_images)
                teacher_labels = teacher(teacher_images, training=False)
            # Freeze layers a subclassed Keras teacher may have built on its first call.
            teacher.trainable = False
            # Dense floating scores are the only supported callable classifier output.
            if not tf.is_tensor(teacher_labels) or isinstance(
                teacher_labels, (tf.RaggedTensor, tf.SparseTensor)
            ) or not tf.as_dtype(teacher_labels.dtype).is_floating:
                raise ValueError("An image-only teacher must return one floating class-score tensor.")

            tf.debugging.assert_rank(
                teacher_labels, 
                2, 
                message="Teacher class scores must have shape [batch, classes]."
            )
            tf.debugging.assert_equal(
                tf.shape(teacher_labels)[0], 
                tf.shape(clean_images)[0], 
                message="Teacher class scores must preserve the image batch size."
            )
            tf.debugging.assert_positive(
                tf.shape(teacher_labels)[1], 
                message="Teacher class scores need at least one class."
            )

            # Softmax and probability checks need more range than mixed-precision inference.
            if teacher_labels.dtype in (tf.float16, tf.bfloat16):
                teacher_labels = tf.cast(teacher_labels, tf.float32)

            tf.debugging.assert_all_finite(teacher_labels, "Teacher class scores must be finite.")
            # Convert raw class scores exactly once, before the shared KD loss.
            if self.teacher_classifier_from_logits:
                teacher_labels = tf.nn.softmax(teacher_labels, axis=-1)
            # Probability teachers must supply an actual distribution per image.
            else:
                tf.debugging.assert_non_negative(
                    teacher_labels, 
                    message="Teacher probabilities must be nonnegative; " 
                    "use teacher_classifier_from_logits=True for logits."
                )
                tf.debugging.assert_near(
                    tf.reduce_sum(teacher_labels, axis=-1), 
                    tf.ones(
                        tf.shape(teacher_labels)[0], 
                        dtype=teacher_labels.dtype
                    ), 
                    atol=1e-3, 
                    rtol=1e-3, 
                    message="Teacher probabilities must sum to one; " 
                    "use teacher_classifier_from_logits=True for logits."
                )
        
        tf.debugging.assert_rank(
            teacher_labels, 
            2, 
            message="Teacher class probabilities must have shape [batch, classes]."
        )

        return tf.stop_gradient(teacher_labels)

    def _classifier_noising_enabled(self, training: bool) -> bool:
        """Whether V1 explicitly overrides the classifier image corruption.

        Args:
            training (bool): Select training cap when true and evaluation cap otherwise.

        Returns:
            enabled (bool): True when noisy classifier input is selected and that
                phase has an explicit corruption cap in constructor configuration.

        Raises:
            This configuration predicate raises no explicit exceptions.
        """

        name = "clf_train_noisified_max_timesteps" if training \
            else "clf_test_noisified_max_timesteps"

        return (
            self.clf_train_noisy_input_type == "noisy"
            and self._init_config[name] is not None
        )

    def _append_classifier_inputs(
        self, 
        prepared: tuple[tf.Tensor, ...], 
        training: bool
    ) -> tuple[tf.Tensor, ...]:
        """Cache capped images/times after diffusion inputs and before teacher targets.

        Inserted images have shape [B,H,W,C] and the clean tensor's floating dtype;
        inserted times are int32 [B]. An enabled nonzero cap advances the named noising
        streams; clean-only [0,0) returns exact clean images without random draws.

        Args:
            prepared (tuple[tf.Tensor, ...]): Seven base diffusion tensors followed
                by optional teacher metadata; element zero is clean ``[B,H,W,C]``.
            training (bool): Select the classifier training or evaluation cap.

        Returns:
            prepared (tuple[tf.Tensor, ...]): Original tuple when no cap is active;
                otherwise adds floating classifier images and int32 times after the
                first seven entries while preserving subsequent teacher metadata.

        Raises:
            ValueError: An enabled classifier corruption cap is rejected by noisify. TensorFlow
                noising errors propagate; a disabled cap does not evaluate or modify the prepared
                tensors.
        """

        # Omitted caps and clean-input training retain the original batch contract.
        if not self._classifier_noising_enabled(training):
            return prepared

        classifier_x, _, classifier_t = self.noisify(
            prepared[0], 
            min_timesteps=0, 
            max_timesteps=self.clf_train_noisified_max_timesteps if training 
                        else self.clf_test_noisified_max_timesteps
        )
    
        return (*prepared[:7], classifier_x, classifier_t, *prepared[7:])

    def _classifier_inputs(
        self, 
        prepared: tuple[tf.Tensor, ...], 
        training: bool
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Select the same cached classifier image/time pair for teacher and student.

        Selection itself consumes no randomness and changes no state. Returned image
        dtype is preserved from the prepared tensor; times are int32 [B] on the clean
        branch and preserve the cached/shared time dtype on the other branches.

        Args:
            prepared (tuple[tf.Tensor, ...]): Base diffusion tuple, optionally with
                classifier images/times cached at positions seven and eight.
            training (bool): True permits shared generator corruption when no cap
                or clean-input policy applies; false otherwise selects clean images.

        Returns:
            inputs (tuple[tf.Tensor, tf.Tensor]): Floating classifier images
                ``[B,H,W,C]`` and integer times ``[B]`` from the chosen cached,
                shared-corruption, or exact-clean path.

        Raises:
            This selector raises no explicit exceptions. A malformed prepared tuple raises
                IndexError when an expected cached position is read.
        """

        # Explicit noisy-input caps have already sampled their shared corruption.
        if self._classifier_noising_enabled(training):
            return prepared[7], prepared[8]

        # Clean training and default evaluation always use exact clean images.
        if not training or self.clf_train_noisy_input_type == "clean":
            return prepared[0], tf.zeros_like(prepared[2], dtype=tf.int32)
        return prepared[3], prepared[2]

    def _prepare_classifier_batch(
        self, 
        inputs: tuple[tf.Tensor, ...], 
        use_label_dropout: bool = True
    ) -> tuple[tuple[tf.Tensor, ...], tf.Tensor | None, tf.Tensor | None]:
        """Separate student inputs, teacher targets, and replay provenance.

        Raw image tensors are numeric external pixels and labels integer [B]. Prepared
        images/noise are floating [B,H,W,C], times int32 [B], condition/class IDs retain
        label dtype and teacher masks are bool [B]. Teacher probabilities are floating
        [B,Ct], or a role-ordered tuple for mapped independent vocabularies. Replay
        provenance retains its incoming bool/numeric [B] dtype. Raw preparation advances
        random streams; decoding an already prepared tuple adds no new corruption.

        Args:
            inputs (tuple[tf.Tensor, ...]): Raw ``(images, classes)`` or
                ``(images, classes, replay_mask)`` data, or their mapped
                seven-tensor equivalents with optional noise-teacher output,
                classifier-teacher target, and final replay mask.
            use_label_dropout (bool): Whether raw input preparation applies
                classifier-free label dropout.
                Defaults to ``True``.

        Returns:
            tuple[tuple[tf.Tensor, ...], tf.Tensor | None, tf.Tensor | None]: Always the
            three-entry outer tuple (prepared_inputs, teacher_labels, replay_mask).
            prepared_inputs contains seven diffusion tensors, a classifier image/time
            pair when capped, and a noise-teacher prediction/mask pair only when noise
            distillation is active. teacher_labels
            is [B, teacher_class_count] when classifier distillation is active, otherwise
            None. replay_mask is the supplied [B] provenance, otherwise None; it is not
            part of prepared_inputs.

        Raises:
            ValueError: Mapped input has an unexpected number of tensors.
        """

        replay_mask = None
        # Map raw supervised pairs or provenance triples before decoding teacher-enabled tensors.
        if self.map_preprocess and len(inputs) in (2, 3):
            self._preprocess_training = use_label_dropout
            inputs = self.prep_inputs_map(*inputs)
            self._preprocess_training = None

        # Mapped data already contains the seven diffusion input tensors.
        if self.map_preprocess:
            expected_length = (
                7 + 2 * int(self.use_noise_distil_loss)
                + int(self.use_classifier_distil)
                + 2 * int(self._classifier_noising_enabled(use_label_dropout))
            )
            # Remove a supplied replay mask before teacher-target extraction.
            if len(inputs) == expected_length + 1:
                inputs, replay_mask = inputs[:-1], inputs[-1]
            # Reject mapped structures that cannot be unambiguously decoded.
            elif len(inputs) != expected_length:
                raise ValueError(
                    "Mapped classifier batches must contain seven student "
                    "tensors, a classifier image/time pair when capped, "
                    "an optional noise-teacher prediction/mask, "
                    "an optional classifier-teacher target, and an optional "
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
                use_label_dropout=use_label_dropout
            )
            prepared_inputs = self._append_classifier_inputs(
                prepared_inputs, 
                use_label_dropout
            )

        # Separate the mapped teacher target from the student input tensors.
        if self.use_classifier_distil:
            prepared_inputs, teacher_labels = (
                prepared_inputs[:-1], 
                prepared_inputs[-1]
            )
        # Keep the ordinary training and evaluation paths teacher-free.
        else:
            teacher_labels = None

        return prepared_inputs, teacher_labels, replay_mask

    def _distillation_metric_mask(
        self, 
        classes: tf.Tensor, 
        teacher_labels: tf.Tensor | None, 
        replay_mask: tf.Tensor | None, 
        classifier_mask: tf.Tensor | None = None
    ) -> tf.Tensor | None:
        """Return the exact rows represented by scoped KD metrics.

        Mapped independent teacher targets may be a tuple of floating [B,Ct] tensors;
        the returned bool [B] mask is the union of role eligibility, intersected with
        classifier selection. No trackers or model state are updated here.

        Args:
            classes (tf.Tensor): Zero-based student targets shaped ``[B]``.
            teacher_labels (tf.Tensor | None): Frozen teacher probabilities;
                their width defines the old-class vocabulary.
            replay_mask (tf.Tensor | None): Per-row replay provenance.
            classifier_mask (tf.Tensor | None): Optional classifier selection
                mask intersected with the KD scope.
                Defaults to ``None``.

        Returns:
            tf.Tensor | None: Boolean row selector, or ``None`` when both the
            KD scope and classifier metrics include every row.

        Raises:
            ValueError: If the configured scope lacks required metadata.
        """

        # Nested targets may represent disjoint, non-prefix vocabularies.
        if isinstance(teacher_labels, (tuple, list)):
            masks = []
            specs = self._classifier_teacher_specs()
            for probabilities, spec in zip(teacher_labels, specs):
                width = tf.cast(self.network.num_classes, tf.int32)
                _, support = self._classifier_teacher_support(probabilities, spec, width)
                masks.append(self._classifier_teacher_mask(
                    classes, 
                    replay_mask, 
                    support, 
                    tf.shape(classes)[0], 
                    self.clf_distil_scope, 
                    role=spec["role"]
                ))

            scope_mask = tf.reduce_any(tf.stack(masks), axis=0)
            # Metric eligibility intersects the same classifier mask as the objective.
            if classifier_mask is not None:
                scope_mask = tf.logical_and(scope_mask, tf.cast(classifier_mask, tf.bool))
            
            return scope_mask

        scope_mask = None
        # Select targets represented by the frozen teacher vocabulary.
        if self.clf_distil_scope == "old_classes":
            # Vocabulary membership cannot be inferred without teacher width.
            if teacher_labels is None:
                raise ValueError(
                    "teacher_labels are required for old_classes metrics."
                )

            scope_mask = tf.reshape(classes, tuple([-1])) < tf.cast(
                tf.shape(teacher_labels)[-1], 
                classes.dtype
            )
        # Select only rows explicitly originating from replay.
        elif self.clf_distil_scope == "replay_only":
            # Replay provenance must be supplied explicitly by the learner.
            if replay_mask is None:
                raise ValueError(
                    "replay_mask is required for replay_only metrics."
                )

            scope_mask = tf.cast(tf.reshape(replay_mask, tuple([-1])), tf.bool)

        # Classifier timestep/CFG filtering also applies to KD measurements.
        if classifier_mask is not None:
            classifier_mask = tf.cast(
                tf.reshape(classifier_mask, tuple([-1])), 
                tf.bool
            )
            # Intersect classifier filtering with a KD scope, or use that filtering as the whole
            # mask.
            scope_mask = classifier_mask if scope_mask is None else tf.logical_and(
                scope_mask, 
                classifier_mask
            )

        return scope_mask

    def _classifier_ctr_metric_mask(
        self, 
        classes: tf.Tensor, 
        teacher_labels: tf.Tensor | None, 
        replay_mask: tf.Tensor | None, 
        classifier_mask: tf.Tensor | None = None
    ) -> tf.Tensor | None:
        """Select the exact rows represented by classifier-token metrics.

        Normal and both modes include every classifier-selected row because the
        ordinary target remains part of the objective. A distil-only regularizer
        additionally intersects the configured KD scope.

        Teacher probabilities may be a mapped tuple of [B,Ct] tensors. Integer class
        IDs and Boolean/numeric masks are [B]. In the ordinary/inactive branch, the
        returned classifier mask retains its supplied dtype; only derived/intersected
        selectors are bool. This method does not update metric accumulators.

        Args:
            classes (tf.Tensor): Dense zero-based student targets shaped [B].
            teacher_labels (tf.Tensor | None): Frozen probabilities [B, teacher_width];
                their width is required for an old_classes distillation-only scope.
            replay_mask (tf.Tensor | None): Replay provenance [B], required for an
                active replay_only distillation-only scope.
            classifier_mask (tf.Tensor | None): Existing classifier selector [B]; None
                includes every row unless the teacher scope narrows it.
                Defaults to ``None``.

        Returns:
            tf.Tensor | None: Combined Boolean selector for distillation-only token
            metrics, or the supplied classifier_mask for normal/both/inactive KD.

        Raises:
            ValueError: Active scope selection lacks its required teacher or replay data.
        """

        # Ordinary token objectives use the classifier mask without adding a KD scope.
        if not self.use_clf_distil_ctr_loss:
            return classifier_mask

        regularizer_kwargs = getattr(
            self.network, 
            "clf_cls_token_regularizer_kwargs", 
            None
        )
        regularizer_kwargs = getattr(
            self.network, 
            "cls_token_regularizer_kwargs", 
            {}
        ) if regularizer_kwargs is None else regularizer_kwargs

        # Normal and blended regularization still include every classifier-selected example.
        if regularizer_kwargs.get("train_type", "normal") != "distil":
            return classifier_mask
        return self._distillation_metric_mask(
            classes, 
            teacher_labels, 
            replay_mask, 
            classifier_mask
        )

    def _classifier_batch_mask(self, classes: tf.Tensor) -> tf.Tensor:
        """Allocate a seeded random subset of rows to classifier training.

        The class tensor is integer [B]; only its leading size is read. Random keys
        are float32 and the returned allocation is bool [B]. This method advances the
        classifier_batch random stream but does not alter labels, weights or metrics.

        Args:
            classes (tf.Tensor): Batch-shaped class targets, used only for row count.

        Returns:
            tf.Tensor: Boolean mask with ``floor(B * fraction)`` selected rows,
            clamped to ``[1, B]``. An empty batch has an empty mask. The dedicated
            stream advances once and participates in reset_seed and checkpoints.

        Raises:
            No explicit exceptions are raised. TensorFlow errors from row-count extraction or the
                stateless random draw propagate.
        """

        batch_size = tf.shape(classes)[0]
        selected_count = tf.minimum(batch_size, tf.maximum(
            1, tf.cast(tf.floor(
                tf.cast(batch_size, tf.float64) * self.clf_train_batch_fraction
            ), tf.int32)
        ))
        # Sorting random keys retains an exact allocation and supports GPU XLA.
        keys = tf.random.stateless_uniform(
            tuple([batch_size]), 
            seed=self._random_streams["classifier_batch"].next_seed()
        )
        permutation = tf.argsort(keys, stable=True)

        return permutation < selected_count

    def _create_metrics(self) -> None:
        """Create classifier trackers while the wrapper is still unbuilt.

        During distillation, class-token metadata selects the primary accuracy
        label. Otherwise it uses ``classifier_accuracy``. Distillation trackers
        are allocated even when their objectives are disabled.

        Args:
            None.

        Returns:
            result (None): Assigns metric objects using the wrapper variable dtype.

        Raises:
            ValueError: A configured Keras metric rejects its constructor options.
        """

        self.ensemble_loss_fn = EnsembleAccuracy(
            self, 
            network_name="raw", 
            max_t=4, 
            seed=self.seed, 
            dtype=self.dtype_policy.variable_dtype
        ) if self.use_ensemble_loss_instead else None

        primary_accuracy_name = "cls_token_accuracy" if getattr(
            self.network, 
            "clf_has_cls_token", 
            False
        ) else "avg_pooling_accuracy"
        configured_distillation = bool(
            self.clf_distil_loss_coef > 0. and 
            getattr(self.network, "distil_token", None) is not None
        )

        stable_dtype = self.dtype_policy.variable_dtype
        self.clf_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="classifier_loss"
        )
        self.clf_kl_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="clf_kl_loss"
        )
        self.clf_ctr_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="clf_ctr_loss"
        )
        self.clf_distil_loss_tracker = metrics.Mean(
            dtype=stable_dtype, 
            name="clf_distil_loss"
        )
        self.previous_teacher_clf_loss_tracker = metrics.Mean(
            dtype=stable_dtype, name="previous_teacher_clf_loss")
        self.current_teacher_clf_loss_tracker = metrics.Mean(
            dtype=stable_dtype, name="current_teacher_clf_loss")
        self.total_accuracy_tracker = metrics.SparseCategoricalAccuracy(
            dtype=stable_dtype, 
            name="total_accuracy"
        )
        self.accuracy_tracker = metrics.SparseCategoricalAccuracy(
            dtype=stable_dtype, 
            name=primary_accuracy_name if configured_distillation
                else "classifier_accuracy"
        )
        self.clf_ctr_accuracy_tracker = metrics.SparseCategoricalAccuracy(
            dtype=stable_dtype, 
            name="clf_ctr_accuracy"
        )
        self.clf_distil_acc_tracker = metrics.SparseCategoricalAccuracy(
            dtype=stable_dtype, 
            name="clf_distil_acc"
        )

    def _reset_classifier_teacher_metrics(self) -> None:
        """Reset accumulated per-role classifier losses after teacher replacement.

        Args:
            None.

        Returns:
            None: Existing previous/current classifier-teacher Mean trackers are reset; absent
                trackers during construction are skipped. Other metrics and all optimizer state are
                retained.

        Raises:
            This helper raises no explicit exceptions; delegated Keras metric reset errors
                propagate.
        """

        for name in ("previous_teacher_clf_loss_tracker", "current_teacher_clf_loss_tracker"):
            tracker = getattr(self, name, None)
            # Constructor attachment may precede classifier metric creation.
            if tracker is not None:
                tracker.reset_state()

    def _validate_classifier_teacher_candidate(
        self, 
        teacher_network: tf.keras.Model | None, 
        role: str, 
        teacher_name: Literal[
            "previous", 
            "current", 
            "classifier"
        ] = "previous"
    ) -> None:
        """Validate class capabilities before replacing or freezing either live teacher.

        A supplied wrapper is inspected through its raw network while its scheduler
        metadata remains available. Disabled roles do not require class outputs.
        Failed validation leaves the current attachments, metrics, and traces intact.

        Args:
            teacher_network (tf.keras.Model | None): Candidate native classifier,
                diffusion wrapper, image-only callable, or None to clear a role.
            role (str): previous or current; selects its loss weight and other role.
            teacher_name (str): Attachment slot being replaced. The classifier slot
                supplies class targets independently of the current noise objective.
                Defaults to ``'previous'``.

        Returns:
            result (None): No attachment mutation; class capabilities and known
                forward-process metadata agree with the enabled objectives.

        Raises:
            ValueError: Required classifier logits/interfaces or metadata are
                incompatible, or clearing a role leaves no required teacher.
        """

        # Base initialization defers classifier-specific validation until its settings exist.
        if not hasattr(self, "clf_distil_loss_coef"):
            return

        regularizer = getattr(self.network, "clf_cls_token_regularizer_kwargs", None)
        # Shared token settings apply only when the classifier has no dedicated settings.
        if regularizer is None:
            regularizer = getattr(self.network, "cls_token_regularizer_kwargs", {})

        needs_classes = bool(self.clf_distil_loss_coef > 0.) or bool(
            self.ctr_loss_coef > 0. and 
            regularizer.get("train_type", "normal") in ("distil", "both")
        )
        # Zero role weights fully disable class KD even when its shared coefficient is positive.
        if not needs_classes or not (
            self.previous_teacher_clf_loss_weight > 0. or 
            self.current_teacher_clf_loss_weight > 0.
        ):
            return

        class_weight = getattr(self, role + "_teacher_clf_loss_weight")
        other_role = "current" if role == "previous" else "previous"
        current = self._current_teacher_spec("classifier")
        other_teacher = (current["network"] if current is not None else None) \
                        if role == "previous" else self.teacher_network
        
        # Clearing the final enabled class teacher is valid only in the deferred lifecycle.
        if teacher_network is None:
            other_active = other_teacher is not None and getattr(
                self, 
                other_role + "_teacher_clf_loss_weight"
            ) > 0.
            # Clearing an unused shared/specialist slot preserves the other current classifier.
            if role == "current":
                other_current = self.classifier_teacher_network if teacher_name == "current" \
                                else self.current_teacher_network
                other_active = other_active or (
                    other_current is not None and 
                    self.current_teacher_clf_loss_weight > 0.
                )

            # A disabled or missing other role cannot supply required class targets.
            if not other_active and not self.defer_teacher:
                raise ValueError("Classifier distillation requires a teacher or defer_teacher=True.")

            return

        # Noise-only roles need no classifier interface.
        if class_weight <= 0.:
            return

        # Deferred attachment must validate student logits before the base setter mutates state.
        if self.clf_distil_loss_coef > 0. or (
            self.ctr_loss_coef > 0. and regularizer.get("train_type", "normal") in ("distil", "both")
        ):
            self._require_student_classifier_logits()

        teacher = teacher_network.network if isinstance(
            teacher_network, 
            DiffusionModel
        ) else teacher_network
        native = callable(getattr(teacher, "predict_class", None))
        # Ordinary image classifiers expose scores through one inference-only call.
        if not native:
            # Native noise outputs and non-callable objects cannot provide class scores.
            if not callable(teacher) or self._teacher_uses_native_noise_api(teacher):
                raise ValueError(
                    "Classification teachers need predict_class or an image-only callable."
                )
            # One plain tensor cannot serve as both class scores and epsilon.
            if teacher_name != "classifier" and self.noise_distil_loss_coef > 0. and getattr(
                self, role + "_teacher_noise_loss_weight"
            ) > 0.:
                raise ValueError(
                    "An image-only class-score teacher requires its noise loss weight to be zero."
                )

            return

        for name in ("timesteps", "channels", "use_cfg"):
            # Task-local vocabulary widths may differ, while diffusion geometry must agree.
            if getattr(teacher, name, None) != getattr(self.network, name, None):
                raise ValueError(f"{role}_teacher_network {name} must match the student.")
        for name, fallback in (
            ("scheduler_name", "_diffusion_scheduler_name"), 
            ("modify_first_t", "_diffusion_modify_first_t")
        ):
            known = getattr(teacher_network, name, getattr(teacher, fallback, None))
            # Native wrappers and snapshots carry optional forward-process metadata.
            if known is not None and known != getattr(self, name):
                raise ValueError(f"{role}_teacher_network {name} must match the student.")

    def _training_optimizers(self) -> tuple:
        """Expose joint and optional phase-specific optimizers for independence checks.

        Args:
            None.

        Returns:
            tuple[tf.keras.optimizers.Optimizer | None, ...]: Entries for optimizer, gen_optimizer
                and clf_optimizer in that order. Uncreated attributes yield None; shared references
                are retained rather than deduplicated. No optimizer is created, reset or updated.

        Raises:
            This attribute accessor raises no explicit exceptions.
        """

        return tuple(getattr(self, name, None) for name in ("optimizer", "gen_optimizer", "clf_optimizer"))

    def _validate_progressive_growth(
        self, 
        model: object, 
        fit_kwargs: Mapping[str, object]
    ) -> None:
        """Preflight native growth and classifier-specific depth constraints on clones.

        Args:
            model (object): Wrapper with a serialization-aware native classifier diffusion network.
            fit_kwargs (Mapping[str, object]): Progressive-fit configuration; requested depth stages
                and classifier requirements are inspected without changing it.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: validate_progressive_classifier_growth rejects an unsupported depth stage or
                incompatible classifier topology. Delegated model reconstruction/build failures
                propagate; live weights are not modified.
        """

        validate_progressive_classifier_growth(model, fit_kwargs)

    def _teacher_fit_methods(self) -> tuple[str, ...]:
        """Extend the base list with native classifier phase-specific entry points.

        Args:
            None.

        Returns:
            tuple[str, ...]: fit, fit_progressively, fit_generator, fit_discriminator and
                fit_generator_progressively, in that order. The dispatcher also checks that the
                selected owner actually implements a callable method.

        Raises:
            This fixed tuple construction raises no explicit exceptions.
        """

        return (
            *super()._teacher_fit_methods(), 
            "fit_generator", 
            "fit_discriminator", 
            "fit_generator_progressively"
        )

    def _teacher_state_attribute(
        self, 
        teacher_name: TeacherName, 
        keras: bool = False
    ) -> str:
        """Resolve the per-role trainer cache or saved Keras fine-tuning-mask attribute.

        Args:
            teacher_name (TeacherName): Supported role identifier.
            keras (bool): Select the ordinary-classifier mask cache when True; False (default)
                selects the training-owner cache.
                Defaults to ``False``.

        Returns:
            str: Previous uses _teacher_model or _keras_teacher_fit_state; other roles insert their
                name after the leading underscore. Validates the role but changes no state.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
        """

        # Native caches retain the base role-local naming contract.
        if not keras:
            return super()._teacher_state_attribute(teacher_name)

        self.get_teacher_network(teacher_name)

        return "_keras_teacher_fit_state" if teacher_name == "previous" else f"_{teacher_name}_keras_teacher_fit_state"

    def _teacher_vocabulary_size(self, network: tf.keras.Model) -> int:
        """Read ordinary output columns or the native conditioning vocabulary.

        Args:
            network (tf.keras.Model): Built single-output image classifier with known
                output_shape[-1], or native diffusion network with num_classes.

        Returns:
            int: Number of ordinary classifier output columns or native real condition classes
                (excluding CFG null). No model evaluation or mutation occurs.

        Raises:
            No explicit exceptions are raised. Reading an unavailable output shape or converting an
                unknown width to int propagates the model/Python error.
        """

        return int(network.output_shape[-1]) if self._uses_keras_teacher_fit(network) \
            else super()._teacher_vocabulary_size(network)

    def _current_teacher_spec(
        self, 
        head: Literal["classifier", "noise"]
    ) -> dict[str, object] | None:
        """Resolve a head-specific current teacher with the corresponding role weight.

        Args:
            head (Literal["classifier", "noise"]): Required head selector; classifier uses
                current_teacher_clf_loss_weight, noise uses current_teacher_noise_loss_weight.

        Returns:
            dict[str, object] | None: Current-role metadata with network, weight, optional
                teacher-column-to-student class_ids and taught task_class_ids; None when no matching
                specialist or shared-current attachment exists.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
        """

        # Classifier loss weights belong only to the classifier wrapper.
        if head == "classifier":
            return self._resolve_current_teacher_spec(
                head, 
                self.current_teacher_clf_loss_weight
            )
        return super()._current_teacher_spec(head)

    def _teacher_model_options(self, network: tf.keras.Model) -> dict[str, object]:
        """Enable supervised native classifier training without recursive distillation.

        Extends the base native options, disables classifier teacher recursion/dynamic
        ordinary-head growth and classifier KD, and replaces a zero classifier loss
        coefficient with one. The outer dictionary is new; network/config objects may
        be shared references. No live model or optimizer is changed.

        Args:
            network (tf.keras.Model): Native classifier diffusion network to wrap.

        Returns:
            dict[str, object]: Native trainer constructor options, including model objects and the
                student dtype policy rather than batch tensors.

        Raises:
            This configuration helper raises no explicit exceptions.
        """

        options = super()._teacher_model_options(network)
        options.update(
            classifier_teacher_network=None, 
            teacher_dynamic_classes=False, 
            clf_loss_coef=options.get("clf_loss_coef", 0.) or 1., clf_distil_loss_coef=0.
        )

        return options

    def _get_teacher_model(
        self, 
        teacher_name: TeacherName = "previous"
    ) -> tf.keras.Model:
        """Use an ordinary classifier itself as its owner; wrap native teachers lazily.

        Caches the ordinary Keras object without cloning or recompiling it, retaining
        its optimizer, loss and fine-tuning configuration. Native roles use the base
        owner construction and resolution synchronization.

        Args:
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.

        Returns:
            tf.keras.Model: The ordinary classifier object or independent native training wrapper.

        Raises:
            ValueError: The role is unknown or has no attached network. Native wrapper construction
                and current-resolution validation also propagate their documented errors.
        """

        network = self.get_teacher_network(teacher_name)

        # Ordinary Keras models keep their compiled graph and optimizer state.
        if self._uses_keras_teacher_fit(network):
            object.__setattr__(
                self, 
                self._teacher_state_attribute(teacher_name), 
                network
            )

            return network
        return super()._get_teacher_model(teacher_name)

    def _compile_teacher(
        self, 
        teacher_name: TeacherName = "previous"
    ) -> None:
        """Preserve ordinary classifiers' existing compile settings and optimizer.

        For image classifiers this only checks optimizer independence. Native roles
        delegate to the base compile lifecycle, which may initialize a separate owner
        from student settings unless compile_teacher supplied explicit settings.

        Args:
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: The role is unknown or has no attached network. Native wrapper construction
                and current-resolution validation also propagate their documented errors.
            ValueError: A selected optimizer is the same object as another training owner's
                optimizer, or shares its iteration variable. Delegated native
                compilation/deserialization errors propagate.
        """

        # Native teacher compilation retains the base independent optimizer lifecycle.
        if not self._uses_keras_teacher_fit(self.get_teacher_network(teacher_name)):
            return super()._compile_teacher(teacher_name)

        self._validate_teacher_optimizers(teacher_name)

    def _validate_trainable_teacher(
        self, 
        network: tf.keras.Model | None, 
        teacher_name: TeacherName
    ) -> None:
        """Check training compatibility without changing the candidate teacher.

        Native models delegate to the base protocol validator. Ordinary image teachers
        need a built graph when training is enabled and cannot provide a positive
        noise objective through the previous/current/noise role. A classifier specialist
        is independent of the student's noise-teaching objective.

        Args:
            network (tf.keras.Model | None): Native diffusion model or wrapper to attach; None
                clears the selected attachment. This is a model object with its own variable dtypes,
                not an image tensor.
            teacher_name (TeacherName): Role identifier used to select previous/current noise loss weight
                and identify the dedicated classifier exception.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: An enabled ordinary training teacher is unbuilt or is assigned a positive
                noise-distillation objective; a native training candidate lacks the required
                diffusion/serialization protocol. Frozen ordinary teachers bypass these training
                requirements.
        """

        # Native diffusion teachers retain the base protocol check.
        if not self._uses_keras_teacher_fit(network):
            return super()._validate_trainable_teacher(network, teacher_name)

        # Unbuilt teachers cannot preserve a complete fine-tuning mask during fit.
        if self.trainable_teacher and not network.built:
            raise ValueError("A trainable Keras classifier teacher must be built.")

        weight = self.previous_teacher_noise_loss_weight if teacher_name == "previous" \
                else self.current_teacher_noise_loss_weight
        
        # Class probabilities cannot also supply epsilon targets.
        if self.trainable_teacher and teacher_name != "classifier" \
        and self.noise_distil_loss_coef > 0. and weight > 0.:
            raise ValueError("A Keras classifier teacher requires its noise loss weight to be zero.")

    def _get_keras_teacher_fit_state(
        self, 
        teacher_name: TeacherName = "previous"
    ) -> tuple | None:
        """Read the original nested fine-tuning mask captured at first attachment.

        Args:
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.

        Returns:
            tuple[tf.keras.Model, tuple[tuple[tf.keras.layers.Layer, bool], ...]] | None: Classifier
                and parent-before-child layer/trainable pairs, or None when no ordinary mask was
                captured. Objects are returned by reference; this does not restore any flags.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
        """

        return getattr(self, self._teacher_state_attribute(teacher_name, keras=True), None)

    def _teacher_label_values(
        self, 
        x: object | None = None, 
        y: object | None = None
    ) -> list[int]:
        """Collect sorted exact sparse IDs from arrays or a finite supervised dataset.

        Dataset discovery eagerly iterates every batch (including an unknown-cardinality
        dataset, which must terminate), ignoring image and optional sample-weight
        values. No teacher mapping or model weights are changed.

        Args:
            x (object | None): Training input; defaults to None. Only a tf.data.Dataset is scanned
                when y is None; each batch must be (images, labels) or (images, labels, weights).
            y (object | None): Integer NumPy-compatible sparse labels [N] or [N,1]. Defaults to
                None, requiring a supervised x dataset. When supplied, y takes precedence; floats,
                booleans and one-hot matrices are rejected.

        Returns:
            list[int]: Sorted unique Python integer IDs; an empty label array/dataset returns an
                empty list.

        Raises:
            ValueError: Labels are missing, the dataset has known infinite cardinality or malformed
                elements, or labels are not one-dimensional/column-vector integers. NumPy/eager
                tensor conversion and dataset iteration errors propagate.
        """

        # Dataset scans must terminate and preserve optional sample-weight tuples.
        if y is None and isinstance(x, tf.data.Dataset):
            cardinality = int(tf.data.experimental.cardinality(x).numpy())
            # Infinite inputs cannot define an exhaustive class vocabulary.
            if cardinality == int(tf.data.INFINITE_CARDINALITY):
                raise ValueError("Dynamic teacher class discovery requires a finite dataset.")
            
            labels = set()
            for batch in x:
                # Keras supervised datasets provide images, labels, and optional weights.
                if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
                    raise ValueError("Dynamic teacher datasets must yield (images, sparse labels[, weights]).")
                
                labels.update(self._teacher_label_values(y=batch[1]))
            
            return sorted(labels)

        # Unlabeled arrays and one-shot generators cannot safely define a class mapping.
        if y is None:
            raise ValueError("Dynamic teacher fitting requires y or a finite supervised tf.data.Dataset.")

        values = np.asarray(y)

        # Column vectors are sparse labels; wider matrices are one-hot or structured targets.
        if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
            raise ValueError("Dynamic teacher fitting requires sparse integer labels.")
        # Do not silently truncate floats or interpret booleans as class IDs.
        if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
            raise ValueError("Dynamic teacher fitting requires sparse integer labels.")
        
        return [int(label) for label in np.unique(values)]

    def _map_teacher_labels(
        self, 
        labels: tf.Tensor, 
        teacher_name: TeacherName = "previous"
    ) -> tf.Tensor:
        """Map sparse dataset IDs to a teacher's persistent classifier columns.

        Requires a previously established _diffusion_seen_classes mapping and leaves
        that mapping and the input tensor unchanged.

        Args:
            labels (tf.Tensor): Integer sparse dataset IDs of shape [N] or [N,1] (other integer
                shapes are preserved). Keys are constructed in this input dtype.
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.

        Returns:
            tf.Tensor: int64 teacher column indices with the same shape as labels; mapping
                comparison/argmax indices are computed internally without casting the input IDs.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
            tf.errors.InvalidArgumentError: At least one label is absent from the established
                teacher vocabulary.
        """

        labels = tf.convert_to_tensor(labels)
        mapping = self.get_teacher_network(teacher_name)._diffusion_seen_classes
        keys = tf.constant(list(mapping), dtype=labels.dtype)
        columns = tf.constant(list(mapping.values()), dtype=tf.int64)
        matches = tf.equal(labels[..., None], keys)
        
        tf.debugging.assert_equal(
            tf.reduce_any(matches, axis=-1), 
            tf.ones_like(labels, dtype=tf.bool), 
            message="Teacher data contains a class not observed during fit."
        )

        return tf.gather(columns, tf.argmax(matches, axis=-1, output_type=tf.int32))

    def _map_teacher_dataset(
        self, 
        dataset: tf.data.Dataset, 
        teacher_name: TeacherName = "previous"
    ) -> tf.data.Dataset:
        """Create a label-remapping dataset while retaining image and weight values.

        Args:
            dataset (tf.data.Dataset): Supervised elements (images, integer labels[, sample
                weights]). Images may have any classifier-compatible numeric dtype/shape; sparse
                labels normally have shape [B] or [B,1].
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.

        Returns:
            tf.data.Dataset: Lazily mapped elements with unchanged image/weight tensors and int64
                labels of the same label shape. Uses num_parallel_calls=1; constructing this
                pipeline does not consume it.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
            tf.errors.InvalidArgumentError: Iterating the result encounters labels absent from the
                selected teacher vocabulary. Dataset structure/tracing errors propagate from
                Dataset.map.
        """

        return dataset.map(
            lambda images, labels, *weights: (images, self._map_teacher_labels(labels, teacher_name), *weights), 
            num_parallel_calls=1
        )

    def _remember_teacher_fit_state(
        self, 
        network: tf.keras.Model | None, 
        teacher_name: TeacherName
    ) -> None:
        """Cache independent training state and preserve an ordinary teacher's layer mask.

        Traverses nested layers parent-first and captures (layer, trainable) pairs only
        on first attachment of that classifier object. Reattaching the same frozen
        object retains its original mask. Replacements invalidate the old owner; native
        or cleared roles discard the ordinary mask. Dynamic ordinary teachers receive
        vocabulary metadata when it is absent. This helper does not itself freeze them.

        Args:
            network (tf.keras.Model | None): Native diffusion model or wrapper to attach; None
                clears the selected attachment. This is a model object with its own variable dtypes,
                not an image tensor.
            teacher_name (TeacherName): Supported role whose owner/mask caches are updated.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
            ValueError: trainable_teacher=True is requested for an unbuilt ordinary Keras
                classifier.
        """

        super()._remember_teacher_fit_state(network, teacher_name)
        cache_attribute = self._teacher_state_attribute(teacher_name)
        state_attribute = self._teacher_state_attribute(teacher_name, keras=True)
        keras_teacher = self._uses_keras_teacher_fit(network)
        state = self._get_keras_teacher_fit_state(teacher_name)

        # Capture parents before children so nested fine-tuning masks restore exactly.
        if network is not None and keras_teacher:
            # An unbuilt classifier has no complete trainability or output contract.
            if self.trainable_teacher and not network.built:
                raise ValueError("A trainable Keras classifier teacher must be built.")

            # Reattaching a frozen teacher preserves the mask from its first attachment.
            if state is None or state[0] is not network:
                layer_states = []


                def remember(layer: tf.keras.layers.Layer) -> None:
                    """Record one nested classifier layer's original trainability.

                    Appends this layer before recursively visiting its direct children in the
                    captured layer_states list; it does not set trainable flags.

                    Args:
                        layer (tf.keras.layers.Layer): Model/layer whose current bool trainable flag and nested
                            direct children are inspected.

                    Returns:
                        None: No value is returned.

                    Raises:
                        This traversal raises no explicit exceptions; errors from a custom layer's child
                            enumeration propagate.
                    """

                    layer_states.append((layer, layer.trainable))
                    for child in layer._flatten_layers(include_self=False, recursive=False):
                        remember(child)


                remember(network)
                object.__setattr__(self, state_attribute, (network, tuple(layer_states)))

            object.__setattr__(self, cache_attribute, network)
            
            # Dynamic classifier identity survives head growth and checkpoint snapshots.
            if self.teacher_dynamic_classes:
                object.__setattr__(network, "_diffusion_dynamic_classes", True)
                # An attached new head discovers its dataset labels on its first fit.
                if not hasattr(network, "_diffusion_seen_classes"):
                    object.__setattr__(network, "_diffusion_seen_classes", {})
        # Native and cleared roles do not own a Keras fine-tuning mask.
        else:
            object.__setattr__(self, state_attribute, None)

    def _check_new_teacher_labels(
        self, 
        x: object | None = None, 
        y: object | None = None, 
        original_labels: Mapping[object, object] | None = None, 
        teacher_class_ids: Sequence[int] | None = None, 
        teacher_name: TeacherName = "previous", 
        verbose: int | bool = True
    ) -> None:
        """Prepare the selected teacher vocabulary before fitting or replay subsampling.

        Native owners delegate discovery/growth to the base method. Fixed ordinary
        classifiers are unchanged. Dynamic ordinary classifiers preserve established
        column identities, append sorted unseen IDs and expand a supported final Dense
        head only when its capacity is insufficient. Existing backbone/head prefixes,
        fine-tuning masks and optimizer-slot prefixes survive supported growth. The
        source and candidate are frozen in a finally block; attachment/cache/vocabulary
        updates already committed before a later error are not transactional.

        Args:
            x (object | None): Finite supervised tf.data.Dataset or raw image input; defaults to
                None. Images are not normalized by label discovery.
            y (object | None): Sparse integer labels [N] or [N,1]; defaults to None, using labels
                from x.
            original_labels (Mapping[object, object] | None): Reporting-only ID-to-display-label
                mapping. None (default) reports original input IDs.
                Defaults to ``None``.
            teacher_class_ids (Sequence[int] | None): Existing output columns' unique dataset IDs,
                in column order. Defaults to None, retaining an established mapping or discovering
                one from training labels. Native/fixed branches leave this option unused here.
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.
            verbose (int | bool): Print newly observed IDs when truthy; defaults to True.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: The role is unknown or has no attached network. Native wrapper construction
                and current-resolution validation also propagate their documented errors.
            ValueError: Sparse labels/class identities are invalid, supplied columns disagree with
                an established mapping, no class is observed, or the ordinary head
                topology/serialized growth is unsupported.
            KeyError: original_labels omits a newly observed ID. Optimizer reconstruction and Keras
                compilation errors propagate after freezing the involved models.
        """

        teacher = self._get_teacher_model(teacher_name)
        # Native condition vocabularies retain base discovery and growth.
        if not self._uses_keras_teacher_fit(teacher):
            return super()._check_new_teacher_labels(
                x=x, y=y, 
                original_labels=original_labels, 
                teacher_name=teacher_name, 
                verbose=verbose
            )

        # Fixed classifiers keep the original output-column label convention.
        if not self.teacher_dynamic_classes:
            return

        width = int(teacher.output_shape[-1])
        mapping = dict(getattr(teacher, "_diffusion_seen_classes", {}))
        # Explicit column identities describe an already trained ordinary head.
        if teacher_class_ids is not None:
            ids = list(teacher_class_ids)
            unique = self._teacher_label_values(y=np.asarray(ids))
            # Every pre-existing output must have one unambiguous dataset identity.
            if len(ids) != width or len(unique) != width:
                raise ValueError("teacher_class_ids must contain one unique integer ID per existing output.")
            
            declared = {int(label): column for column, label in enumerate(ids)}
            # A fitted teacher's established column order cannot be reinterpreted.
            if mapping and mapping != declared:
                raise ValueError("teacher_class_ids conflicts with the existing teacher vocabulary.")
            
            mapping = declared

        labels = self._teacher_label_values(x=x, y=y)
        new_labels = [label for label in labels if label not in mapping]
        reported = new_labels if original_labels is None else [original_labels[label] for label in new_labels]
        for label in new_labels:
            mapping[label] = len(mapping)

        # Empty data cannot establish a usable sparse output vocabulary.
        if not mapping:
            raise ValueError("Dynamic teacher fitting requires at least one observed class.")

        layer_states = self._get_keras_teacher_fit_state(teacher_name)[1]
        candidate = teacher
        try:
            for layer, trainable in layer_states:
                layer.trainable = trainable

            candidate = expand_classifier_head(teacher, max(width, len(mapping)))
            # A resized graph needs a compatible optimizer with preserved old moment prefixes.
            if candidate is not teacher:
                compile_config = tf.keras.utils.deserialize_keras_object(teacher.get_compile_config())
                compile_config["optimizer"] = register_optimizer_variables(
                    teacher.optimizer, 
                    candidate.trainable_variables, 
                    preserve_slot_prefixes=True
                )
                candidate.compile(**compile_config)
            object.__setattr__(candidate, "_diffusion_seen_classes", mapping)
            object.__setattr__(candidate, "_diffusion_dynamic_classes", True)
            self._attach_fitted_teacher(candidate, teacher_name)
        finally:
            # Both the retained source and a failed candidate remain inference-only.
            teacher.trainable = False
            candidate.trainable = False

        # Report only genuinely new classes, using the caller's dataset names when supplied.
        if verbose and new_labels:
            print("The teacher has found new Classes:", reported)

    @property
    def metrics(self) -> list[metrics.Metric]:
        """Return diffusion and classifier metric trackers.

        Args:
            None.

        Returns:
            list[tf.keras.metrics.Metric]: Base diffusion trackers followed by
            classifier losses and classifier/combined/token accuracies.
            Available after :meth:`compile`.

        Raises:
            This accessor raises no explicit exceptions; it returns active existing metric objects
                without resetting their values.
        """

        return [
            *super().metrics, 
            self.clf_loss_tracker, 
            self.clf_kl_loss_tracker, 
            self.clf_ctr_loss_tracker, 
            self.clf_distil_loss_tracker, 
            *([self.previous_teacher_clf_loss_tracker, self.current_teacher_clf_loss_tracker]
            if self._uses_mapped_classifier_teachers() else []), 
            self.total_accuracy_tracker, 
            self.accuracy_tracker, 
            self.clf_ctr_accuracy_tracker, 
            self.clf_distil_acc_tracker
        ]

    def compile(self, **kwargs: object) -> None:
        """Compile the wrapper and create classifier metrics/loss helper.

        Args:
            **kwargs (object): Forwarded compile options, empty by default. loss defaults to
                "mse"
                through DiffusionModel; omitting optimizer uses Keras
                optimizer="rmsprop". Other supported keys include run_eagerly,
                steps_per_execution, jit_compile where available, metrics,
                weighted_metrics, and loss_weights, with omitted Keras defaults retained.

        Returns:
            None: Configures base losses and all base/classifier trackers, including
            distillation trackers even when their objectives are disabled. Installs KL
            divergence and uses the constructor's four-timestep raw-network
            EnsembleAccuracy helper when use_ensemble_loss_instead is enabled.
            Recompilation refreshes objective/logits flags and resets
            accumulated tracker state.

        Raises:
            ValueError: A selected optimizer is the same object as another training owner's
                optimizer, or shares its iteration variable.
            Delegated DiffusionModel/Keras compilation or teacher compilation errors propagate.
                Successful earlier state changes are not rolled back.
        """

        self._refresh_loss_flags()
        super().compile(**kwargs)

        self.kld_loss_fn = losses.kullback_leibler_divergence

    def train_step(
        self, 
        inputs: tuple[tf.Tensor, ...]
    ) -> dict[str, tf.Tensor]:
        """Perform one joint raw-network diffusion/classifier update.

        Raw batches contain numeric external images [B,H,W,C] (or [B,H,W] before
        channel insertion) and sparse integer labels [B]. Prepared batch layout/dtypes
        follow the relevant prep_inputs/prep_clfv2_inputs adapter; classifier variants
        can carry bool/numeric replay provenance [B] and floating per-role teacher
        targets. Returned metric values are scalar tensors in their trackers' policy
        variable dtype. Raw preparation advances the appropriate saved random streams;
        prepared tensors reuse their already sampled corruption.

        Args:
            inputs (tuple[tf.Tensor, ...]): Clean images and zero-based classes,
                optionally followed by a replay mask, or the prepared tensors
                supplied by ``map_preprocess``.

        Returns:
            dict[str, tf.Tensor]: Running diffusion and classifier metrics.  The
            classifier mask selects CFG-null and/or low-timestep examples when
            configured; divide-no-nan makes an empty selection contribute zero.
            Clean/noisy and CFG/null classifier inputs are independent of those
            row masks. A positive batch fraction allocates disjoint classifier
            and diffusion rows within one primary prediction; zero preserves
            the full-batch denoising objective and optional classifier prediction.

        Raises:
            ValueError: Raw/mapped batch structure or active teacher/loss metadata is invalid.
                TensorFlow/Keras errors from network execution, loss reduction, gradient application
                or metrics propagate; completed updates are not rolled back.
        """

        prepared_inputs, teacher_labels, replay_mask = (
            self._prepare_classifier_batch(inputs)
        )
        classifier_x, classifier_t = self._classifier_inputs(
            prepared_inputs, 
            training=True
        )
        # Extract cached noise-teacher predictions and vocabulary masks before student computation.
        if self.use_noise_distil_loss:
            teacher_noises_pred = prepared_inputs[-2]
            teacher_noise_mask = prepared_inputs[-1]
            prepared_inputs = prepared_inputs[:-2]
        # Teacher-free training has no noise-distillation targets or teacher-vocabulary mask.
        else:
            teacher_noises_pred = None
            teacher_noise_mask = None 
        (x0, noises, 
        t, x_t, 
        cfg_labels, 
        uncond_labels, 
        classes) = prepared_inputs[:7]

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        clf_loss_mask = tf.ones_like(
            cfg_labels, 
            dtype=stable_dtype
        )
        # Restrict classifier metrics and loss to CFG-dropped examples.
        if self.mask_by_nulls:
            clf_loss_mask = clf_loss_mask * tf.cast(
                cfg_labels == 0, 
                dtype=stable_dtype
            )
        # Restrict classifier metrics and loss to the configured leading timesteps.
        if self.mask_by_t_threshold:
            clf_loss_mask = clf_loss_mask * tf.cast(
                t <= self.filter_t_threshold, 
                dtype=stable_dtype
            )

        clean_classifier = self.clf_train_noisy_input_type == "clean"
        capped_classifier = self._classifier_noising_enabled(training=True)
        null_classifier = self.clf_train_class_input_type == "null_class_only"
        split_batch = self.clf_train_batch_fraction > 0.
        diffusion_mask = None
        forward_x, forward_t, forward_labels = x_t, t, cfg_labels

        # Allocate disjoint objectives before changing the selected classifier inputs.
        if split_batch:
            classifier_mask = self._classifier_batch_mask(classes)
            diffusion_mask = tf.logical_not(classifier_mask)
            clf_loss_mask *= tf.cast(classifier_mask, stable_dtype)

            # Apply the selected image corruption only to allocated classifier rows.
            if clean_classifier or capped_classifier:
                forward_x = tf.where(
                    classifier_mask[:, None, None, None], 
                    classifier_x, 
                    x_t
                )
                forward_t = tf.where(
                    classifier_mask, 
                    classifier_t, 
                    t
                )

            # Null conditioning changes allocated classifier rows without changing row masks.
            if null_classifier:
                forward_labels = tf.where(
                    classifier_mask, 
                    uncond_labels, 
                    cfg_labels
                )

        clf_acc_mask = tf.cast(
            clf_loss_mask, 
            dtype=tf.bool
        )

        separate_classifier = not split_batch and (clean_classifier or capped_classifier or null_classifier)
        with tf.GradientTape() as tape:
            forward_outputs = self.forward(
                "raw", forward_x, forward_t, forward_t, 
                cond_labels=forward_labels, 
                uncond_labels=uncond_labels, 
                scale=self.train_cfg_scale, 
                return_logits=bool(self.use_logits_instead), 
                training=True
            )
            (x0_pred, noises_pred, 
            (regs_list_c, regs_list_u), 
            (z_vals_list_c, z_vals_list_u), 
            (classes_pred_c, classes_pred_u), 
            (clf_regs_list_c, clf_regs_list_u), 
            (clf_z_vals_list_c, clf_z_vals_list_u)) = forward_outputs[:7]

            # The logits pair belongs to the forward pass already performed above.
            if self.use_logits_instead:
                logits_c, logits_u = forward_outputs[-1]
            # Paths without KD or token losses keep their original tuple shape.
            else:
                logits_c, logits_u = ({}, {})

            # The forward pass appends the selected distillation predictions.
            if self.use_clf_distil_loss:
                distil_classes_c, distil_classes_u = forward_outputs[7]
            # Disabled distillation has no associated predictions.
            else:
                distil_classes_c = None
                distil_classes_u = None

            (loss1, noise_loss, cond_noise_loss, 
            uncond_noise_loss, noise_distil_loss, 
            image_loss, kl_loss1, ctr_loss1, 
            ctr_preds1) = self.compute_batch_diffusion_losses(
                diffusion_mask, 
                x0=x0, noises=noises, classes=classes, 
                x0_pred=x0_pred, noises_pred=noises_pred, 
                z_vals_list_c=z_vals_list_c, 
                regs_list_c=regs_list_c, 
                z_vals_list_u=z_vals_list_u, 
                regs_list_u=regs_list_u, 
                cond_labels=cfg_labels, 
                teacher_noises_pred=teacher_noises_pred, 
                teacher_noise_mask=teacher_noise_mask
            )
            # Replace only classifier outputs; denoising keeps its original forward pass.
            if separate_classifier:
                # Both selectors compose into one call on the shared trainable backbone.
                class_outputs = self.network.predict_class(
                    (
                        classifier_x, 
                        classifier_t, 
                        uncond_labels if null_classifier else cfg_labels
                    ), 
                    max_encoder_num=None, 
                    full_return=True, 
                    training=True, 
                    **self.use_logits_instead
                )
                classes_pred_c = class_outputs[0]
                clf_regs_list_c = class_outputs[3]
                clf_z_vals_list_c = class_outputs[4]
                distil_classes_c = class_outputs[
                    5 if getattr(self.network, "distil_token", None) is not None else 0
                ] if self.use_clf_distil_loss else None
                logits_c = class_outputs[-1] if self.use_logits_instead else {}

            (loss2, clf_loss, kl_loss2, 
            ctr_loss2, clf_distil_loss, 
            classes_pred, ctr_preds2, 
            distil_classes) = self.compute_clf_kl_ctr_distil_loss(
                classes, classes_pred_c, 
                clf_z_vals_list_c, clf_regs_list_c, 
                distil_classes_c=distil_classes_c, 
                logits_c=logits_c, 
                classes_pred_u=classes_pred_u, 
                clf_z_vals_list_u=clf_z_vals_list_u, 
                clf_regs_list_u=clf_regs_list_u, 
                distil_classes_u=distil_classes_u, 
                logits_u=logits_u, 
                clf_loss_mask=clf_loss_mask, 
                clf_train_type="cond", 
                kl_train_type="cond" if separate_classifier or split_batch else None, 
                ctr_train_type="cond" if separate_classifier or split_batch else None, 
                teacher_labels=teacher_labels, 
                replay_mask=replay_mask, 
                x0=x0, 
                training=True
            )

            loss = loss1 + loss2

        self.apply_grads(tape, loss)
        self.update_ema()

        diffusion_classes, diffusion_labels = classes, cfg_labels
        metric_teacher_noise_mask = teacher_noise_mask
        # Diffusion means are weighted only by rows that contributed to their objectives.
        if split_batch:
            diffusion_classes = tf.boolean_mask(classes, diffusion_mask)
            diffusion_labels = tf.boolean_mask(cfg_labels, diffusion_mask)
            metric_teacher_noise_mask = tf.nest.map_structure(
                lambda mask: tf.boolean_mask(mask, diffusion_mask), 
                teacher_noise_mask
            ) if teacher_noise_mask is not None else None

        results = self.get_results_dict(
            noise_loss, 
            cond_noise_loss=cond_noise_loss, 
            uncond_noise_loss=uncond_noise_loss, 
            noise_distil_loss=noise_distil_loss, 
            teacher_noise_mask=metric_teacher_noise_mask, 
            total_loss=loss, 
            image_loss=image_loss, 
            kl_loss=kl_loss1, 
            ctr_loss=ctr_loss1, 
            ctr_preds=ctr_preds1, 
            classes=diffusion_classes, 
            cond_labels=diffusion_labels, 
            use_total_loss=not split_batch
        )

        # The joint objective remains one batch-level update even when either allocation is empty.
        if split_batch:
            self.total_loss_tracker.update_state(
                loss, 
                sample_weight=tf.cast(tf.shape(classes)[0], stable_dtype)
            )
            results[self.total_loss_tracker.name] = self.total_loss_tracker.result()

        # Compute a regularizer metric scope only when classifier token loss is active. Compute a KD
        # metric scope only when the independent distillation objective is active.
        results.update(self.get_clf_results_dict(
            clf_loss, 
            classes, 
            classes_pred, 
            clf_acc_mask, 
            clf_kl_loss=kl_loss2, 
            clf_ctr_loss=ctr_loss2, 
            clf_distil_loss=clf_distil_loss, 
            clf_ctr_preds=ctr_preds2, 
            distil_classes=distil_classes, 
            use_total_loss=False, 
            clf_ctr_mask=self._classifier_ctr_metric_mask(
                classes, 
                teacher_labels, 
                replay_mask, 
                clf_acc_mask
            ) if self.use_clf_ctr_loss else None, 
            clf_distil_acc_mask=self._distillation_metric_mask(
                classes, 
                teacher_labels, 
                replay_mask, 
                clf_acc_mask
            ) if self.use_clf_distil_loss else None
        ))

        return results

    def test_step(
        self, 
        inputs: tuple[tf.Tensor, ...]
    ) -> dict[str, tf.Tensor]:
        """Evaluate diffusion plus unconditional clean or capped-image classification.

        Diffusion metrics use noisified inputs and the configured test CFG scale.
        Classifier metrics call ``predict_class`` with null labels on every row.
        Images are clean at timestep zero unless noisy inputs and an explicit
        classifier test cap select bounded noising. Training masks do not
        restrict evaluation.

        Raw batches contain numeric external images [B,H,W,C] (or [B,H,W] before
        channel insertion) and sparse integer labels [B]. Prepared batch layout/dtypes
        follow the relevant prep_inputs/prep_clfv2_inputs adapter; classifier variants
        can carry bool/numeric replay provenance [B] and floating per-role teacher
        targets. Returned metric values are scalar tensors in their trackers' policy
        variable dtype. Raw preparation advances the appropriate saved random streams;
        prepared tensors reuse their already sampled corruption.

        Args:
            inputs (tuple[tf.Tensor, ...]): Clean images and zero-based classes,
                optionally followed by a replay mask, or the prepared tensors
                supplied by ``map_preprocess``.

        Returns:
            dict[str, tf.Tensor]: Running enabled diffusion and classifier
            losses/accuracies.

        Raises:
            ValueError: Raw/mapped batch structure or required teacher/loss metadata is invalid.
                TensorFlow/Keras prediction, reduction and metric errors propagate. No optimizer
                update is performed.
        """

        prepared_inputs, teacher_labels, replay_mask = (
            self._prepare_classifier_batch(
                inputs, 
                use_label_dropout=False
            )
        )
        classifier_x, classifier_t = self._classifier_inputs(
            prepared_inputs, 
            training=False
        )
        # Extract the noise-teacher pair from mapped evaluation batches when enabled.
        if self.use_noise_distil_loss:
            teacher_noises_pred = prepared_inputs[-2]
            teacher_noise_mask = prepared_inputs[-1]
            prepared_inputs = prepared_inputs[:-2]
        # Teacher-free evaluation supplies no noise target or teacher mask.
        else:
            teacher_noises_pred = None
            teacher_noise_mask = None  
        (x0, noises, 
        t, x_t, 
        cond_labels, 
        uncond_labels, 
        classes) = prepared_inputs[:7]

        (loss1, noise_loss, cond_noise_loss, 
        uncond_noise_loss, noise_distil_loss, 
        image_loss, kl_loss1, ctr_loss1, 
        ctr_preds1) = super().forward_and_compute_loss(
            self.test_network_name, 
            x0, noises, t, x_t, 
            cond_labels=cond_labels, 
            uncond_labels=uncond_labels, 
            classes=classes, 
            cfg_scale=self.test_cfg_scale, 
            teacher_noises_pred=teacher_noises_pred, 
            teacher_noise_mask=teacher_noise_mask, 
            # kl_train_type="cond", 
            # ctr_train_type="cond", 
            use_image_loss=True, 
            training=False
        )

        # Evaluation keeps null conditioning with the selected classifier corruption.
        class_outputs = self.get_network(self.test_network_name).predict_class(
            (classifier_x, classifier_t, uncond_labels), 
            max_encoder_num=None, 
            full_return=True, 
            training=False, 
            **self.use_logits_instead
        )

        classes_pred = class_outputs[0]
        clf_regs_list = class_outputs[3]
        clf_z_vals_list = class_outputs[4]
        distil_classes = class_outputs[
            5 if getattr(self.network, "distil_token", None) is not None else 0
        ] if self.use_clf_distil_loss else None
        logits = class_outputs[-1] if self.use_logits_instead else None

        (loss2, clf_loss, kl_loss2, 
        ctr_loss2, clf_distil_loss, 
        classes_pred, ctr_preds2, 
        distil_classes) = self.compute_clf_kl_ctr_distil_loss(
            classes, None, None, None, None, 
            classes_pred, clf_z_vals_list, 
            clf_regs_list, distil_classes, 
            clf_train_type="uncond", 
            kl_train_type="uncond", 
            ctr_train_type="uncond", 
            teacher_labels=teacher_labels, 
            x0=x0, 
            replay_mask=replay_mask, 
            logits_u=logits, 
            training=False
        )

        loss = loss1 + loss2

        results = self.get_results_dict(
            noise_loss, 
            cond_noise_loss=cond_noise_loss, 
            uncond_noise_loss=uncond_noise_loss, 
            noise_distil_loss=noise_distil_loss, 
            teacher_noise_mask=teacher_noise_mask, 
            total_loss=loss, 
            image_loss=image_loss, 
            kl_loss=kl_loss1, 
            ctr_loss=ctr_loss1, 
            ctr_preds=ctr_preds1, 
            classes=classes, 
            cond_labels=cond_labels, 
            use_image_loss=True
        )
        # Scope token metrics only when the classifier regularizer objective is active. Scope
        # distillation metrics only when the independent teacher objective is active.
        results.update(self.get_clf_results_dict(
            clf_loss, 
            classes, 
            classes_pred, 
            clf_kl_loss=kl_loss2, 
            clf_ctr_loss=ctr_loss2, 
            clf_distil_loss=clf_distil_loss, 
            clf_ctr_preds=ctr_preds2, 
            distil_classes=distil_classes, 
            use_total_loss=False, 
            clf_ctr_mask=self._classifier_ctr_metric_mask(
                classes, 
                teacher_labels, 
                replay_mask
            ) if self.use_clf_ctr_loss else None, 
            clf_distil_acc_mask=self._distillation_metric_mask(
                classes, 
                teacher_labels, 
                replay_mask
            ) if self.use_clf_distil_loss else None
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
                        "vision_transformer_block": True
                    }
                ]
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

        Raises:
            No new validation is introduced here. The base progressive trainer propagates
                stage/growth, callback and fitting errors. Completed stages and structural growth
                remain; this override adds classifier-depth history only after successful delegated
                fitting.
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
            dataset (tf.data.Dataset): Finite clean external-pixel [B,H,W,C] batches
                (or grayscale [B,H,W]) paired with sparse dataset labels. Images
                pass through prepare_images for conversion and active-resolution resizing.
                In replay_only scope a third tensor is KD provenance and is
                stripped; otherwise a third tensor is interpreted as sample weights.
            verbose (bool): Print batch progress when true.
                Defaults to ``True``.
            **kwargs (object): EnsembleAccuracy overrides, empty by default. Defaults here
                use
                test_network_name, max_t=min(128, timesteps), wrapper clf_acc_coef,
                wrapper seed, and policy variable dtype. Regularizer/distillation
                coefficients default to their wrapper values only while those losses
                are active, otherwise zero. Compute mode, weighting, chunk size, and
                separate_probas retain EnsembleAccuracy constructor defaults.
                prediction_batch_size defaults to None, using each input batch's
                size before timestep and CFG expansion as the classifier call
                limit. An explicit integer overrides this automatic limit.

        Returns:
            float: Sparse categorical accuracy across the full dataset.

        Raises:
            ValueError: The combined primary-head coefficient is nonfinite. The delegated ensemble
                evaluator also rejects unsupported estimator settings and invalid input/label
                contracts. Errors raised while mapping a dataset or running a network propagate.
        """

        # Replay-only continual datasets use their third tensor as KD
        # provenance, not as an accuracy sample weight. Remove that metadata
        # before the generic ensemble metric interprets Keras triples.
        element_spec = getattr(dataset, "element_spec", None)
        # Strip replay metadata only from the documented replay-only triple.
        if self.clf_distil_scope == "replay_only" \
        and isinstance(element_spec, (tuple, list)) \
        and len(element_spec) == 3:
            def _drop_replay_provenance(
                images: tf.Tensor, 
                labels: tf.Tensor, 
                replay_mask: tf.Tensor
            ) -> tuple[tf.Tensor, tf.Tensor]:
                """Remove KD-only provenance before generic accuracy evaluation.

                The input tensor objects are returned by reference without normalization,
                casting, copying, iteration or metric/random/model-state updates.

                Args:
                    images (tf.Tensor): Numeric image batch, normally [B,H,W,C] (or [B,H,W]
                        before channel insertion), in the dataset's external coordinates.
                    labels (tf.Tensor): Sparse integer class IDs [B] or [B,1], preserving
                        their incoming integer dtype.
                    replay_mask (tf.Tensor): Boolean/numeric provenance [B], deliberately
                        omitted from the returned accuracy input pair.

                Returns:
                    tuple[tf.Tensor, tf.Tensor]: The unchanged images and labels, retaining
                    exactly their input shapes and dtypes.

                Raises:
                    This tuple adapter raises no explicit exceptions.
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
        # Include default regularizer-head weight only when classifier regularizers are active.
        kwargs.setdefault(
            "ctr_acc_coef", 
            self.ctr_acc_coef if self.use_clf_ctr_loss else 0.
        )
        # Include default distillation weight only when its teacher objective is active.
        kwargs.setdefault(
            "clf_distil_acc_coef", 
            self.clf_distil_acc_coef if self.use_clf_distil_loss else 0.
        )
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("dtype", self.dtype_policy.variable_dtype)

        ensemble_accuracy = EnsembleAccuracy(
            self, 
            **kwargs
        )
        # Fold validated scalar weights when both predictions use the primary head.
        if getattr(self.network, "distil_token", None) is None:
            ensemble_accuracy.clf_acc_coef += ensemble_accuracy.clf_distil_acc_coef
            # Folding finite inputs can still overflow their combined coefficient.
            if not np.isfinite(ensemble_accuracy.clf_acc_coef):
                raise ValueError("Combined primary-head accuracy coefficient must be finite.")
            ensemble_accuracy.clf_distil_acc_coef = 0.
        accuracy_value = ensemble_accuracy.evaluate(
            dataset, 
            verbose=verbose
        )

        return float(accuracy_value)

    def get_teacher_names(self) -> tuple[TeacherName, ...]:
        """List classifier-wrapper teacher roles in lifecycle order.

        Args:
            None.

        Returns:
            tuple[TeacherName, ...]: ``("previous", "current", "classifier", "noise")``. The tuple contains
                selector strings, not model objects; no state changes.

        Raises:
            This fixed tuple accessor raises no explicit exceptions.
        """

        return ("previous", "current", "classifier", "noise")

    def get_teacher_network(
        self, 
        teacher_name: TeacherName = "previous"
    ) -> tf.keras.Model | None:
        """Read an attached classifier, noise, shared-current or previous teacher.

        Args:
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.

        Returns:
            tf.keras.Model | None: Existing raw network/image classifier for the role, or None for a
                valid empty role. Native wrappers are represented by their raw attached networks; no
                owner is constructed.

        Raises:
            ValueError: The role name is not one of get_teacher_names().
        """

        return super().get_teacher_network(teacher_name)

    def get_teacher_model(
        self, 
        teacher_name: TeacherName = "previous"
    ) -> tf.keras.Model:
        """Get the cached independent training owner for the selected role.

        Args:
            teacher_name (TeacherName): Role returned by get_teacher_names; the default ``"previous"``
                selects the historical teacher slot. A valid unattached role is distinct from an
                invalid name.

        Returns:
            tf.keras.Model: Ordinary image classifiers are returned unchanged with their own
                compilation. Native teachers use a lazily created independent wrapper
                (DiffusionModel for noise); native resolution is synchronized. No fitting occurs.

        Raises:
            ValueError: The role is unknown or has no attached network. Native wrapper construction
                and current-resolution validation also propagate their documented errors.
        """

        return self._get_teacher_model(teacher_name)

    def set_classifier_teacher_network(
        self, 
        network: tf.keras.Model | None, 
        class_ids: Sequence[int] | None = None, 
        task_class_ids: Sequence[int] | None = None
    ) -> None:
        """Attach, replace or clear the current-task classifier specialist.

        Delegates to the classifier-aware current setter, freezes the model, remembers
        its original fine-tuning mask, resets role metrics and invalidates student
        traces. Does not train the attachment or copy its weights.

        Args:
            network (tf.keras.Model | None): Native diffusion model or wrapper to attach; None
                clears the selected attachment. This is a model object with its own variable dtypes,
                not an image tensor.
            class_ids (Sequence[int] | None): One student class ID per output column; -1 marks
                unsupported columns. Defaults to None, using persistent dynamic metadata or the
                implicit native/leading-column convention.
            task_class_ids (Sequence[int] | None): Student IDs taught by this specialist; None
                (default) uses mapped/available support.
                Defaults to ``None``.

        Returns:
            None: No value is returned.

        Raises:
            ValueError: Shared-current and specialist attachments conflict, the candidate cannot
                supply classifier outputs or trainable metadata, or class/taught-support mappings
                are invalid. Objective-validation assertions propagate.
        """

        self.set_current_teacher_network(
            network, 
            class_ids, 
            task_class_ids, 
            teacher_name="classifier"
        )

    def fit_teacher(
        self, 
        x: object | None = None, 
        y: object | None = None, 
        fit_method: Literal[
            "fit", 
            "fit_progressively", 
            "fit_generator", 
            "fit_discriminator", 
            "fit_generator_progressively"
        ] = "fit", 
        teacher_class_ids: Sequence[int] | None = None, 
        teacher_name: TeacherName = "previous", 
        **kwargs: object
    ) -> callbacks.History | dict[str, list]:
        """Fit only the teacher through its native wrapper or compiled Keras model.

        Select one independent training owner with teacher_name. A native noise
        specialist uses DiffusionModel; a native classifier teacher uses its
        classifier wrapper, and an ordinary Keras classifier keeps its own fit.
        Newly created native trainers reuse constructor noising, CFG, input and
        loss settings, excluding distillation, EMA and auxiliary image/KL/token
        losses. Supplied native wrappers retain their own configuration. Zero
        supervised coefficients become one for a newly created native trainer.

        A built, compiled ordinary Keras classifier uses only fit_method="fit".
        Compile it with compile_teacher or attach an already compiled model.
        Its own optimizer, loss and metrics are retained across student compile
        and repeated teacher fits. The layer trainability recorded at attachment
        is restored for each fit, including frozen backbone and BatchNormalization
        layers; the teacher is frozen again even if fitting fails. Inputs,
        images, sample weights and callbacks retain their ordinary Keras meanings.
        Fixed teachers take output-column label IDs; dynamic teachers take sparse
        dataset IDs and remap training/validation labels and class_weight keys.
        For get_model(..., model_type="pretrained"), use raw [0,255] images.
        Distillation restores clean images with public postprocess before
        calling an ordinary classifier teacher.
        With teacher_dynamic_classes=True, sparse dataset IDs are mapped to a
        persistent vocabulary, and the final Dense head grows without shrinking.
        New IDs append in sorted order. Declare an already trained head's original
        output-column identities with teacher_class_ids on its first fit.
        Dynamic teachers support automatic continual training; native progressive,
        generator, and discriminator fit methods remain unsupported for image teachers.

        For ordinary array inputs, x is normally numeric [N,H,W,C] and y an integer
        [N] or [N,1] sparse target; images keep their incoming dtype. Dynamic remapping
        produces int64 targets with the same label shape. None x/y omit separate inputs
        for native delegation; ordinary model.fit receives those None values directly.
        Teacher role metrics and student execution caches refresh on reattachment.
        Invalid validation vocabularies are rejected before head growth; later failures
        can retain already committed vocabulary/head/optimizer or training updates.

        Args:
            x (object | None): Clean images or a finite (images, labels) dataset.
                Native teachers apply their wrapper preprocessing; ordinary Keras teachers
                receive x unchanged in their own expected coordinates.
                Defaults to ``None``.
            y (object | None): Optional separate sparse labels, as in fit.
                Defaults to ``None``.
            fit_method (str): fit (default), or fit_progressively for the existing
                depth/timestep/resolution curriculum. Fixed and dynamic class counts
                use that method's normal label discovery and optimizer growth.
                V2 also accepts its existing fit_generator, fit_discriminator, and
                fit_generator_progressively methods for phase-specific delegation.
                Defaults to ``'fit'``.
            teacher_class_ids (Sequence[int] | None): Optional original dataset IDs
                in existing Keras output-column order. Requires dynamic teacher mode;
                cannot change an established vocabulary. None discovers the first
                vocabulary from supplied training labels in sorted order.
                Defaults to None; the existing saved vocabulary is retained when present, otherwise
                first-seen training IDs establish sorted columns.
            teacher_name (TeacherName):
                Select teacher_network (default), the shared current teacher, or one
                current head specialist. Only that teacher is fitted or expanded.
                Defaults to ``'previous'``.
            **kwargs (object): Passed to the selected teacher fit method.
                Native V2 teachers use their existing gen_kwargs/clf_kwargs instead
                of separate x/y. Teacher and student must share label-ID meanings
                for subsequent distillation, as with any runtime teacher.

        Returns:
            callbacks.History | dict[str, list]: The delegated teacher history
            (a mapping for V2). Cached optimizer/vocabulary state survives repeated
            calls. Student weights and optimizer are unchanged.

        Raises:
            ValueError: Teacher training is disabled, the role is unknown/empty, the student or
                ordinary teacher is uncompiled, fit_method is unsupported, optimizer state aliases
                another owner, or supplied dynamic-head/validation label identities are invalid.
                teacher_class_ids is rejected outside dynamic ordinary-classifier fitting.
            KeyError: class_weight or reporting original_labels addresses an unknown class. Errors
                from supported head expansion, Keras fitting and callbacks propagate; the selected
                teacher is frozen again after its fit attempt.
        """

        # Explicit opt-in preserves the existing frozen-teacher behavior.
        if not self.trainable_teacher or self.get_teacher_network(teacher_name) is None:
            raise ValueError("fit_teacher requires trainable_teacher=True and the selected teacher.")
        # Compilation establishes the independent teacher's loss and optimizer settings.
        if not self.compiled:
            raise ValueError("Call compile before fit_teacher.")

        # Ordinary image classifiers train directly with their existing compile state.
        if self._uses_keras_teacher_fit(self.get_teacher_network(teacher_name)):
            # Progressive and phase-specific methods require a native diffusion teacher.
            if fit_method != "fit":
                raise ValueError("A Keras classifier teacher supports only fit_method='fit'.")

            # Uncompiled image teachers require their own classification objective.
            if not getattr(self.get_teacher_network(teacher_name), "compiled", False):
                raise ValueError("Call compile_teacher before fit_teacher for a Keras classifier.")

            self._compile_teacher(teacher_name)
            # Dynamic teachers discover only training IDs and reject unseen validation targets.
            if self.teacher_dynamic_classes:
                discovered = self._teacher_label_values(x=x, y=y)
                known = set(getattr(self.get_teacher_network(teacher_name), "_diffusion_seen_classes", {}))
                # An explicitly declared pretrained vocabulary is also valid for validation.
                if teacher_class_ids is not None:
                    known.update(teacher_class_ids)

                known.update(discovered)
                validation = kwargs.get("validation_data")
                # Validation is never allowed to create new head columns.
                if validation is not None:
                    validation_labels = self._teacher_label_values(x=validation) if isinstance(
                        validation, tf.data.Dataset
                    ) else self._teacher_label_values(y=validation[1])
                    # Reject unknown validation IDs before growth or an optimizer update.
                    if set(validation_labels) - known:
                        raise ValueError("Teacher validation contains a class not observed during fit.")
                
                self._check_new_teacher_labels(
                    y=np.asarray(discovered, dtype=np.int64), 
                    original_labels=kwargs.pop("original_labels", None), 
                    teacher_class_ids=teacher_class_ids, 
                    teacher_name=teacher_name, 
                    verbose=kwargs.get("verbose", True)
                )
                # Dataset labels and optional weights retain their batch alignment.
                if isinstance(x, tf.data.Dataset):
                    x = self._map_teacher_dataset(x, teacher_name)
                # Array targets retain their sparse vector/column-vector shape.
                else:
                    y = self._map_teacher_labels(y, teacher_name)
                # Map validation with the completed training vocabulary only.
                if isinstance(validation, tf.data.Dataset):
                    kwargs["validation_data"] = self._map_teacher_dataset(validation, teacher_name)
                # Validation tuples may carry optional sample weights.
                elif validation is not None:
                    kwargs["validation_data"] = (
                        validation[0], 
                        self._map_teacher_labels(validation[1], teacher_name), 
                        *validation[2:]
                    )
                # Keras class weights address teacher output columns after remapping.
                if kwargs.get("class_weight") is not None:
                    mapping = self.get_teacher_network(teacher_name)._diffusion_seen_classes
                    kwargs["class_weight"] = {
                        mapping[label]: weight for label, 
                        weight in kwargs["class_weight"].items()
                    }
            # Fixed teachers retain exact forwarding and need no identity declaration.
            elif teacher_class_ids is not None:
                raise ValueError("teacher_class_ids requires teacher_dynamic_classes=True.")

            teacher, layer_states = self._get_keras_teacher_fit_state(teacher_name)
            try:
                for layer, trainable in layer_states:
                    layer.trainable = trainable
                teacher.make_train_function(force=True)

                return teacher.fit(x=x, y=y, **kwargs)
            finally:
                # Freeze before validation and discard student traces containing old targets.
                teacher.trainable = False
                self._attach_fitted_teacher(teacher, teacher_name)

        # Native teacher wrappers already carry their own class vocabulary.
        if teacher_class_ids is not None:
            raise ValueError("teacher_class_ids applies only to dynamic Keras classifier teachers.")

        return super().fit_teacher(
            x=x, y=y, 
            fit_method=fit_method, 
            teacher_name=teacher_name, 
            **kwargs
        )

    def compile_teacher(
        self, 
        teacher_name: TeacherName = "previous", 
        **kwargs: object
    ) -> None:
        """Configure only the teacher's optimizer, loss, metrics and execution options.

        Requires trainable_teacher=True and an attached teacher. This method may
        run before student compile, but fit_teacher still requires a compiled
        student. The teacher remains frozen outside its own compile/fit calls.

        Ordinary Keras classifiers receive their original fine-tuning mask and
        all options unchanged. Supply the classification loss and metrics, e.g.
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]. Native
        diffusion teachers use their wrapper's compile defaults, including MSE.
        Explicit teacher settings survive later student compile calls; replacing
        the teacher clears the native override. Recompilation follows the
        teacher's usual Keras optimizer and metric reset semantics.

        A supplied optimizer must also be independent of every other attached teacher,
        not only the student. Native owner creation may allocate variables. Ordinary
        classifier fine-tuning flags are restored only for compile, then frozen again
        in a finally block. This method performs no batch updates.

        Args:
            teacher_name (TeacherName):
                Attached teacher to compile; previous preserves the original API.
                Defaults to ``'previous'``.
            **kwargs (object): Forwarded to the teacher's compile method. Use a
                separate optimizer instance or name; teacher and student must
                not share an optimizer, including either V2 phase optimizer.

        Returns:
            None: Teacher compilation changes; student weights, optimizer, 
            metrics and cached execution functions are preserved.

        Raises:
            ValueError: Teacher training is disabled, no teacher is attached, 
                or the requested optimizer shares student update state.
        """

        # Public compilation is available only for an explicitly trainable teacher.
        if not self.trainable_teacher or self.get_teacher_network(teacher_name) is None:
            raise ValueError(
                "compile_teacher requires trainable_teacher=True and the selected teacher."
            )

        self._check_teacher_optimizer(kwargs.get("optimizer"), self)
        for other_name in self.get_teacher_names():
            # Current specialists must not share optimizer state with each other either.
            if other_name != teacher_name and self.get_teacher_network(other_name) is not None:
                self._check_teacher_optimizer(
                    kwargs.get("optimizer"), 
                    self.get_teacher_model(other_name)
                )

        # Fine-tuned image classifiers own their original nested trainability mask.
        if self._uses_keras_teacher_fit(self.get_teacher_network(teacher_name)):
            teacher, layer_states = self._get_keras_teacher_fit_state(teacher_name)
            
            try:
                for layer, trainable in layer_states:
                    layer.trainable = trainable

                teacher.compile(**kwargs)
            finally:
                # Compilation failures must also leave the inference teacher frozen.
                teacher.trainable = False
            
            return

        return super().compile_teacher(teacher_name=teacher_name, **kwargs)

    def snapshot_teacher_network(
        self, 
        network_name: NetworkName | Literal["teacher"] = "raw"
    ) -> tf.keras.Model:
        """Clone a prediction copy into an independent frozen raw teacher.

        The clone is built, assigned the active resolution, and populated by matching
        active layer names so dynamic class/depth replacement cannot scramble weights.
        Forward-process metadata is attached for subsequent compatibility checks.
        Ordinary Keras classifiers retain their architecture, weights and dataset
        vocabulary instead of diffusion-specific metadata. The snapshot is returned
        without installing it on the wrapper.

        Args:
            network_name (NetworkName | Literal["teacher"]): raw or ema student
                branch, or teacher for the currently attached independent teacher. An ema request falls
                back to raw when use_ema=False, as in get_network.
                Defaults to ``'raw'``.

        Returns:
            tf.keras.Model: Independent raw-network clone with trainable=False and the
            selected weights, current topology, resolution, preprocessing, and schedule metadata.

        Raises:
            ValueError: The branch is unknown or cloned layers cannot match every weight.
            Exception: Delegated serialization, construction, or shape failures propagate.
        """

        source_network = self.get_network(network_name)
        # Ordinary classifier snapshots retain their independent graph and dataset vocabulary.
        if self._uses_keras_teacher_fit(source_network):
            teacher_network = models.clone_model(source_network)
            teacher_network.set_weights(source_network.get_weights())
            for name in (
                "_diffusion_seen_classes", 
                "_diffusion_dynamic_classes", 
                "_diffusion_task_class_ids"
            ):
                # Copy only metadata actually owned by this source.
                if hasattr(source_network, name):
                    value = getattr(source_network, name)
                    object.__setattr__(
                        teacher_network, 
                        name, 
                        dict(value) if isinstance(value, dict) else value
                    )

            teacher_network.trainable = False

            return teacher_network
        
        return super().snapshot_teacher_network(network_name)

    def forward( # call function
        self, 
        network_name: NetworkName, 
        x_t: tf.Tensor, 
        t: tf.Tensor, 
        t_batch: tf.Tensor, 
        cond_labels: tf.Tensor, 
        uncond_labels: tf.Tensor | None = None, 
        scale: float | None = None, 
        return_logits: bool = False, 
        training: bool | None = None
    ) -> tuple[
        tf.Tensor, tf.Tensor, 
        tuple[list[tf.Tensor], list[tf.Tensor] | None], 
        tuple[
            list[tuple[tf.Tensor, tf.Tensor]], 
            list[tuple[tf.Tensor, tf.Tensor]] | None
        ]
    ]:
        """Run network pass(es), guidance, and algebraic x0 reconstruction.

        Reconstructed x0 and epsilon are floating [B,H,W,C]; reconstruction follows
        denoise's dtype rule, while auxiliary probabilities/logits [B,K] and latent
        pairs [B,...] preserve native output dtypes. Classifier forwards retain the
        conditional/null auxiliary pairs from call_network, with optional distillation
        and logits pairs in its documented order. A noise-only teacher follows the
        base four-entry forward contract. training=None delegates mode resolution to
        Keras/native layers; forward itself applies no optimizer or metric update.

        Args:
            network_name (NetworkName | Literal["teacher"]): Existing raw, EMA, or runtime
                teacher branch selected through get_network.
            x_t (tf.Tensor): Noisy images ``[B,H,W,C]``.
            t (tf.Tensor): Scalar or batch timestep used to gather schedule rates.
            t_batch (tf.Tensor): Batch-shaped integer timesteps ``[B]`` supplied
                to the network's embedding.
            cond_labels (tf.Tensor): Conditional label IDs ``[B]``.
            uncond_labels (tf.Tensor | None): Null labels for CFG.
                Defaults to ``None``.
            scale (float | None): Guidance scale; None skips unconditional pass.
                Defaults to ``None``.
            return_logits (bool): Request additive classifier-logit metadata from
                a compatible classifier call_network implementation. Defaults to False.
            training (bool | None): Keras training mode. Defaults to ``None``.

        Returns:
            tuple: ``(x0, eps, (regs_c, regs_u),
            (z_vals_list_c, z_vals_list_u))``. Image tensors
            match ``x_t``; regularizers and latent pairs preserve branch outputs.

        Raises:
            ValueError: Network selection is unsupported or the requested teacher is absent. Native
                network, schedule and guided-reconstruction errors propagate; this helper does not
                apply gradients.
        """

        network_options = {"return_logits": True} if return_logits else {}

        (eps_c, eps_u), *others = self.call_network(
            x_t, 
            t_batch, 
            cond_labels, 
            uncond_labels, 
            scale, 
            network_name=network_name, 
            training=training, 
            **network_options
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

    def _uses_keras_teacher_fit(self, teacher: tf.keras.Model | None) -> bool:
        """Identify ordinary Keras image classifiers without probing their outputs.

        Args:
            teacher (tf.keras.Model | None): Candidate model or absent attachment. Native wrappers
                and objects with predict_class/full_return are excluded.

        Returns:
            bool: True only for a Keras Model that is neither a DiffusionModel nor a native
                classifier/noise teacher. Does not require compile/build state and makes no
                mutation.

        Raises:
            No explicit exceptions are raised. The native-noise signature helper treats
                uninspectable signatures as non-native.
        """

        return (
            isinstance(teacher, tf.keras.Model)
            and not isinstance(teacher, DiffusionModel)
            and not callable(getattr(teacher, "predict_class", None))
            and not self._teacher_uses_native_noise_api(teacher)
        )

    def set_teacher_network(self, teacher_network: tf.keras.Model | None) -> None:
        """Install a native or image-only teacher and refresh classifier objectives.

        Base attachment unwraps raw teacher weights, rejects student aliases, applies
        a frozen inference state, and resets execution caches. Once classifier coefficients exist,
        this override validates predict_class and forward-process compatibility for
        teacher-dependent classifier/token objectives. Runtime teachers are not saved
        in the student's normal serialized configuration.

        Args:
            teacher_network (tf.keras.Model | None): Independent raw teacher or diffusion
                wrapper. None clears the teacher when deferred attachment permits it;
                positive teacher objectives otherwise require an attached teacher.

        Returns:
            None: Updates classifier/noise loss flags and combined-accuracy availability.
            Active teacher objectives force mapped preprocessing; clearing them restores
            the original map_preprocess choice. During base construction, classifier
            validation waits until classifier coefficients have been installed.

        Raises:
            ValueError: Required teacher predictions, known schedule/timestep conventions,
                channels, or CFG metadata are incompatible, or base attachment fails.
        """

        self._validate_classifier_teacher_candidate(teacher_network, "previous")
        super().set_teacher_network(teacher_network)

        # Base construction installs teachers before classifier coefficients are available.
        if not hasattr(self, "clf_distil_loss_coef"):
            return

        self._reset_classifier_teacher_metrics()
        self._refresh_loss_flags()
        self.map_preprocess = True if (
            self.use_noise_distil_loss or 
            self.use_classifier_distil
        ) else self._map_preprocess_without_teacher

    def set_current_teacher_network(
        self, 
        teacher_network: tf.keras.Model | None, 
        class_ids: list[int] | tuple[int, ...] | None = None, 
        task_class_ids: list[int] | tuple[int, ...] | None = None, 
        teacher_name: Literal["current", "classifier", "noise"] = "current"
    ) -> None:
        """Validate and attach a frozen current teacher with independent class mappings.

        Args:
            teacher_network (tf.keras.Model | None): Independent native or image-only
                classifier, diffusion wrapper, or None to clear the current role.
            class_ids (list[int] | tuple[int, ...] | None): Student class IDs in
                teacher-column order; None uses the inherited leading-ID convention.
                Defaults to ``None``.
            task_class_ids (list[int] | tuple[int, ...] | None): Taught subset of
                class_ids; None inherits all mapped classes.
                Defaults to ``None``.
            teacher_name (str): current selects the legacy shared expert; classifier
                or noise selects an independent current-task specialist.
                Defaults to ``'current'``.

        Returns:
            result (None): Attaches/freezes the role, resets class/noise teacher
                metrics, refreshes objective flags, and invalidates cached batch graphs.

        Raises:
            ValueError: Classifier capabilities or inherited teacher/mapping
                requirements conflict with the student objectives.
        """

        # Noise specialists need no classifier interface or class-only capability checks.
        if teacher_name != "noise":
            self._validate_classifier_teacher_candidate(
                teacher_network, 
                "current", 
                teacher_name
            )

        super().set_current_teacher_network(
            teacher_network, 
            class_ids, 
            task_class_ids, 
            teacher_name
        )

        # Base construction attaches roles before classifier settings exist.
        if not hasattr(self, "clf_distil_loss_coef"):
            return

        self._reset_classifier_teacher_metrics()
        self._refresh_loss_flags()
        self.map_preprocess = True if (
            self.use_noise_distil_loss or 
            self.use_classifier_distil
        ) else self._map_preprocess_without_teacher

    def prep_inputs_map(
        self, 
        x0: tf.Tensor, 
        labels: tf.Tensor, 
        replay_mask: tf.Tensor | None = None
    ) -> tuple[tf.Tensor, ...]:
        """Prepare one input-pipeline batch and optional teacher targets.

        Args:
            x0 (tf.Tensor): Clean image batch.
                Numeric raw external images [B,H,W,C] or [B,H,W]; preprocessing converts them to model
                coordinates and policy variable dtype before optional resizing.
            labels (tf.Tensor): Dataset class labels.
                Sparse integer dataset IDs [B]; integer dtype is preserved by dynamic mapping before
                network-specific casting.
            replay_mask (tf.Tensor | None): Optional per-example replay
                provenance. The continual learner supplies this as the third
                dataset tensor for ``clf_distil_scope="replay_only"``.
                Defaults to ``None``.

        Returns:
            tuple[tf.Tensor, ...]: The seven values from :meth:`prep_inputs`,
            a classifier image/time pair when capped, then an optional
            noise-teacher prediction/mask and frozen
            classifier-teacher probabilities. A supplied replay mask remains
            the final tensor.

        Raises:
            ValueError: Raw preparation, noising or teacher inference rejects its configured
                contract. TensorFlow label-map and teacher shape/numerics assertion failures
                propagate during mapping.
        """

        # Retain the training CFG scale for combined targets, including an unset mode sentinel.
        if self.use_noise_distil_loss and self.use_classifier_distil \
        and self._preprocess_training is not False \
        and self._teacher_uses_native_noise_api() \
        and self._current_teacher_spec("classifier") is None \
        and self._current_teacher_spec("noise") is None:
            prepared_inputs = self.prep_inputs(
                (x0, labels), 
                use_label_dropout=True
            )
            prepared_inputs = self._append_classifier_inputs(
                prepared_inputs, 
                training=True
            )
            x_t = prepared_inputs[3]
            t = prepared_inputs[2]
            cond_labels = prepared_inputs[4]
            uncond_labels = prepared_inputs[5]
            teacher_noise_mask = tf.ones_like(
                cond_labels, 
                dtype=tf.bool
            )
            teacher_num_labels = getattr(
                self.teacher_network, 
                "num_labels", 
                None
            )
            
            # Mask new-class noise targets when the teacher has a narrower condition vocabulary.
            if teacher_num_labels is not None:
                teacher_noise_mask = cond_labels < tf.cast(
                    teacher_num_labels, 
                    cond_labels.dtype
                )

            teacher_outputs = self.forward(
                "teacher", x_t, t, t, 
                cond_labels=self._mask_unknown_teacher_labels(cond_labels), 
                uncond_labels=uncond_labels, 
                scale=self.train_cfg_scale, 
                training=False
            )
            teacher_noises_pred = tf.stop_gradient(teacher_outputs[1])
            
            # The default classifier uses the already-computed primary teacher output.
            if self.clf_train_noisy_input_type == "noisy" \
            and self.clf_distil_noisy_input_type == "noisy" \
            and self.clf_distil_train_noisified_max_timesteps is None \
            and self.clf_train_class_input_type == "all_classes" \
            and not self._classifier_noising_enabled(training=True):
                teacher_labels = teacher_outputs[4][0]
            # Every other combination needs one teacher prediction on the selected inputs.
            else:
                classifier_x, classifier_t = self._classifier_inputs(prepared_inputs, training=True)
                teacher_labels = self._predict_teacher_labels(
                    classifier_x, 
                    classifier_t, 
                    uncond_labels if self.clf_train_class_input_type == "null_class_only"
                    else self._mask_unknown_teacher_labels(cond_labels), 
                    clean_images=prepared_inputs[0], 
                    training=True
                )

            prepared_inputs = (
                *prepared_inputs, 
                teacher_noises_pred, 
                teacher_noise_mask, 
                tf.stop_gradient(teacher_labels)
            )

            # Append provenance only when the caller supplied a replay mask.
            return prepared_inputs if replay_mask is None else (
                *prepared_inputs, 
                replay_mask
            )

        prepared_inputs = super().prep_inputs_map(x0, labels)
        training = self._preprocess_training is not False
        prepared_inputs = self._append_classifier_inputs(prepared_inputs, training)

        # Preserve the ordinary mapped batch when no teacher target is used.
        if not self.use_classifier_distil:
            # Preserve optional provenance even when no classifier-teacher target is required.
            return prepared_inputs if replay_mask is None else (
                *prepared_inputs, 
                replay_mask
            )

        teacher_x, teacher_t = self._classifier_inputs(prepared_inputs, training)

        # Validation uses null labels; teacher caps may override the selected student inputs.
        if self._preprocess_training is False:
            teacher_labels_in = prepared_inputs[5] # uncond_labels
        # Compose the same independent input selectors used by the student classifier.
        else:
            teacher_labels_in = prepared_inputs[
                5 if self.clf_train_class_input_type == "null_class_only" else 4
            ] # cfg_labels or uncond_labels

        # Map only unseen condition IDs to null for narrower past teachers.
        # The dual helper maps the original condition independently for each teacher.
        if self._current_teacher_spec("classifier") is None:
            teacher_labels_in = self._mask_unknown_teacher_labels(teacher_labels_in)
        
        teacher_labels = self._predict_teacher_labels(
            teacher_x, 
            teacher_t, 
            teacher_labels_in, 
            clean_images=prepared_inputs[0], 
            training=training
        )
        teacher_inputs = (*prepared_inputs, teacher_labels)

        # Keep supplied provenance last, after all precomputed teacher targets.
        return teacher_inputs if replay_mask is None else (
            *teacher_inputs, replay_mask
        )

    def call_network(
        self, 
        x_t: tf.Tensor, 
        t_batch: tf.Tensor, 
        cond_labels: tf.Tensor, 
        uncond_labels: tf.Tensor | None = None, 
        scale: float | None = None, 
        network_name: NetworkName = "raw", 
        return_logits: bool = False, 
        training: bool = False
    ) -> tuple[tuple[object, object | None], ...]:
        """Run conditional and optional unconditional ``DiTClassifier`` passes.

        Native image/noise outputs are floating [B,H,W,C], class and regularizer
        probabilities/logits are floating [B,K], and latent pairs contain matching
        floating [B,...] tensors. Each preserves its producing layer's dtype. The
        conditional/null pairs can have None for an unused null branch. A requested
        noise-only teacher instead returns the base three-pair noise protocol. The
        extra logits metadata contains class_logits, distil_logits and
        clf_regs_logits_list; absent optional entries remain None. Native training
        mode may advance stochastic/normalization state; no optimizer step occurs.

        Args:
            x_t (tf.Tensor): Noisy images ``[B,H,W,C]``.
            t_batch (tf.Tensor): Integer timesteps ``[B]``.
            cond_labels (tf.Tensor): Conditional/possibly dropped labels ``[B]``.
            uncond_labels (tf.Tensor | None): Null labels ``[B]``.
                Defaults to ``None``.
            scale (float | None): Non-None requests an unconditional pass when
                CFG is enabled; combination is performed by inherited denoise.
                Defaults to ``None``.
            network_name (NetworkName | Literal["teacher"]): raw, ema, or attached teacher
                branch. A noise-only teacher uses the
                base DiffusionModel three-pair output contract.
                Defaults to ``'raw'``.
            return_logits (bool): Append conditional/unconditional same-pass logits metadata
                for active soft distillation. Defaults to ``False``.
            training (bool): Keras training mode.
                Defaults to ``False``.

        Returns:
            tuple[tuple[object, object | None], ...]: Six conditional/
            unconditional pairs in order: noise tensors, main regularizer
            lists, main latent-statistic pairs, classifier probabilities,
            classifier regularizer lists, and classifier latent-statistic
            pairs. Student distillation mode appends one distillation-token
            probability pair; teachers supply their primary class probabilities
            without requiring a distillation head. Unconditional members are None when no second pass is
            requested.
            ``return_logits=True`` additionally appends a same-pass logits pair
            for active soft KD. Ordinary tuple positions and lengths are unchanged.

        Raises:
            ValueError: Network selection is unsupported or a requested teacher is absent. Native
                classifier output-key/shape errors and noise-only teacher adapter errors propagate;
                this method does not validate arbitrary custom native dictionaries itself.
        """

        # A noise-only teacher follows the base denoiser contract without classifier outputs.
        if network_name == "teacher" and not self.use_classifier_distil:
            return DiffusionModel.call_network(
                self, 
                x_t, 
                t_batch, 
                cond_labels, 
                uncond_labels, 
                scale, 
                network_name=network_name, 
                training=training
            )

        network = self.get_network(network_name)

        # Frozen teachers need probabilities only; students expose same-pass loss logits.
        logits_options = self.use_logits_instead if return_logits and network_name != "teacher" \
                        else {}
        # The combined previous-teacher pass needs the same placement as separate targets.
        with self._teacher_inference_scope(network) if network_name == "teacher" else nullcontext():
            output_dict_c = network(
                (x_t, t_batch, cond_labels), 
                full_return=True, 
                training=training, 
                **logits_options
            )
            output_dict_u = network(
                (x_t, t_batch, uncond_labels), 
                full_return=True, 
                training=training, 
                **logits_options
            ) if network.use_cfg and scale is not None else {}

        eps_c, eps_u = output_dict_c["noises"], output_dict_u.get("noises")
        regs_list_c, regs_list_u = output_dict_c["regs_list"], output_dict_u.get("regs_list")
        z_vals_list_c, z_vals_list_u = output_dict_c["z_vals_list"], output_dict_u.get("z_vals_list")
        classes_pred_c, classes_pred_u = output_dict_c["classes"], output_dict_u.get("classes")
        clf_regs_list_c, clf_regs_list_u = output_dict_c["clf_regs_list"], output_dict_u.get("clf_regs_list")
        clf_z_vals_list_c, clf_z_vals_list_u = output_dict_c["clf_z_vals_list"], output_dict_u.get("clf_z_vals_list")

        outputs = (
            (eps_c, eps_u), 
            (regs_list_c, regs_list_u), 
            (z_vals_list_c, z_vals_list_u), 
            (classes_pred_c, classes_pred_u), 
            (clf_regs_list_c, clf_regs_list_u), 
            (clf_z_vals_list_c, clf_z_vals_list_u)
        )

        # Frozen targets use primary probabilities and need no student distillation head.
        if self.use_clf_distil_loss and network_name != "teacher":
            distil_key = "distil_classes" if getattr(self.network, "distil_token", None) is not None else "classes"
            outputs += tuple([(
                output_dict_c[distil_key], 
                output_dict_u.get(distil_key)
            )])

        # Append only logits metadata, preserving every existing probability tuple position.
        if logits_options:
            keys = ("class_logits", "distil_logits", "clf_regs_logits_list")
            outputs += tuple([({key: output_dict_c.get(key) for key in keys}, 
                         {key: output_dict_u.get(key) for key in keys})])

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

        Sparse targets use integer dtype; probabilities and optional image context are
        real-floating [B,K] and [B,H,W,C]. Cross-entropy/reduction and numeric mask casts
        use policy variable dtype, and the scalar returned loss has that dtype. The
        returned probability tensor preserves its producer's dtype. None for x0 is
        unused without an ensemble; None training is forwarded to the ensemble Keras
        execution context. This method does not update trackers or optimizers.

        Args:
            classes (tf.Tensor): Zero-based targets shaped ``[B]``.
            classes_pred (tf.Tensor): Primary class probabilities shaped
                ``[B,num_classes]``.
            clf_loss_mask (tf.Tensor | None): Optional float sample weights
                shaped ``[B]``.
                Defaults to ``None``.
            x0 (tf.Tensor | None): Clean images required by ensemble loss.
                Defaults to ``None``.
            training (bool | None): Mode forwarded to ensemble prediction.
                Defaults to ``None``.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Scalar loss and the primary or
            ensemble probabilities used to compute it.

        Raises:
            No explicit exceptions are raised here. Ensemble prediction, sparse categorical
                cross-entropy and mask broadcasting/reduction errors propagate. With a zero mask,
                divide_no_nan returns zero for finite per-row losses.
        """

        # The optional ensemble supplies classifier probabilities for this loss.
        if self.ensemble_loss_fn is not None:
            classes_pred = self.ensemble_loss_fn.ensemble_predict_batched(
                x0, 
                training=training
            )

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        stable_predictions = tf.cast(classes_pred, stable_dtype)
        clf_loss = tf.cast(
            self.scce_loss_fn(classes, stable_predictions), 
            stable_dtype
        )
        clf_loss_mask = tf.cast(clf_loss_mask, stable_dtype) \
                        if clf_loss_mask is not None else None

        clf_loss = tf.math.divide_no_nan(
            tf.reduce_sum(clf_loss * clf_loss_mask), 
            tf.reduce_sum(clf_loss_mask)
        ) if clf_loss_mask is not None else tf.reduce_mean(clf_loss)

        return clf_loss, classes_pred

    def compute_clf_distil_loss(
        self, 
        teacher_labels: tf.Tensor, 
        distil_classes: tf.Tensor, 
        clf_distil_type: Literal["hard", "soft"] | None = None, 
        clf_distil_loss_mask: tf.Tensor | None = None, 
        classes: tf.Tensor | None = None, 
        replay_mask: tf.Tensor | None = None, 
        clf_distil_temperature: float | None = None, 
        clf_distil_scope: Literal[
            "old_classes", 
            "replay_only", 
            "current_and_replay"
        ] | None = None, 
        kd_loss_allocator: Callable | None = None, 
        x0: tf.Tensor | None = None, 
        student_logits: tf.Tensor | None = None, 
        teacher_loss_weight: float | None = None, 
        update_teacher_metrics: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute hard-label CE or soft-label KL distillation loss.

        The teacher and student share their leading class-ID vocabulary but may have
        different widths. Teacher probabilities are truncated to student width,
        renormalized on retained support, then zero-padded for newly added student
        classes. Temperature softening preserves exact zero probabilities and uses
        log-space normalization before padding, so unseen classes receive no invented
        teacher mass. Eligible rows require positive retained teacher mass. Soft KD
        uses same-pass student logits, log-softmax,
        and T**2 scaling at every positive temperature.
        The scope and classifier mask are multiplied and the loss is divided by their
        total weight; an empty mask produces zero. Teacher targets are stop-gradient.

        Args:
            teacher_labels (tf.Tensor): Frozen teacher probabilities [B, teacher_width] on
                the common leading
                class-ID support; teacher_width may differ from student_width.
            distil_classes (tf.Tensor): Selected student-head probabilities [B,
                student_width], returned
                unchanged alongside the policy-variable-dtype scalar loss.
            clf_distil_type (Literal["hard", "soft"] | None): Loss mode;
                ``None`` uses the wrapper's token-distillation setting.
                Defaults to ``None``.
            clf_distil_loss_mask (tf.Tensor | None): Optional float sample weights
                shaped ``[B]``.
                Defaults to ``None``.
            classes (tf.Tensor | None): Sparse zero-based targets used to
                identify examples from teacher-known classes.
                Defaults to ``None``.
            replay_mask (tf.Tensor | None): Boolean/float replay provenance
                used only by the ``"replay_only"`` scope.
                Defaults to ``None``.
            clf_distil_temperature (float | None): Finite positive soft-KD temperature; None
                inherits the configured value.
                Hard KD ignores softening but still validates this parameter.
                Defaults to ``None``.
            clf_distil_scope (Literal["old_classes", "replay_only", "current_and_replay"] | None):
                Exposure scope; None inherits the configured scope. old_classes needs
                classes and uses the original teacher width; replay_only needs explicit
                replay_mask; current_and_replay adds no extra row filter.
                Defaults to ``None``.
            kd_loss_allocator (Callable | None): Optional runtime independent-head
                KD reducer. Receives this wrapper, the unchanged per-example KD,
                its final eligibility mask, clean x0, classes, and replay provenance.
                The aggregation method supplies it only during training.
                None (default) uses the standard eligible-row mean; a callable receives (wrapper,
                per-row variable-dtype loss [B], optional weights [B], x0, classes, replay_mask) and
                returns the reduced scalar loss.
                Defaults to ``None``.
            x0 (tf.Tensor | None): Clean normalized images for an optional reducer.
                None (default) supplies no clean-image context to a custom allocator; otherwise floating
                model-coordinate images [B,H,W,C] are passed unchanged.
                Defaults to ``None``.
            student_logits (tf.Tensor | None): Same-pass unnormalized student scores.
                Required for saturated probabilities. Direct callers may omit them
                only when all supplied probabilities are strictly positive.
                Defaults to ``None``.
            teacher_loss_weight (float | None): Multiplier for a single teacher
                term; None uses previous_teacher_clf_loss_weight. Dual-role
                aggregation instead uses each role's configured coefficient.
                None (default) selects previous_teacher_clf_loss_weight for the legacy single-teacher
                route and each configured weight for mapped roles.
                Defaults to ``None``.
            update_teacher_metrics (bool): True updates independent per-role
                trackers for tuple targets. Single-tensor direct calls leave
                role trackers unchanged. Defaults to False.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Scalar role-weighted distillation loss
            in policy variable dtype and the unchanged student probabilities.
            The outer clf_distil_loss_coef is applied by the loss aggregator.

        Raises:
            ValueError: Loss mode, temperature or scope is invalid, or the selected
                scope lacks its required class labels or replay mask.
            tf.errors.InvalidArgumentError: Teacher/student support is empty, or
                a row selected by the final scope and mask has no retained teacher mass.
        """

        clf_distil_type = self.clf_distil_type if clf_distil_type is None else clf_distil_type
        clf_distil_temperature = self.clf_distil_temperature if clf_distil_temperature is None \
                                else clf_distil_temperature
        clf_distil_scope = self.clf_distil_scope if clf_distil_scope is None else clf_distil_scope

        # Unknown loss modes must not silently select a different scientific objective.
        if clf_distil_type not in ("hard", "soft"):
            raise ValueError("clf_distil_type must be 'hard' or 'soft'.")
        # Zero or negative temperature would invalidate probability softening.
        if not np.isfinite(clf_distil_temperature) or clf_distil_temperature <= 0.:
            raise ValueError(
                "clf_distil_temperature must be finite and positive."
            )
        # Reject unknown sample-selection policies before tensor computation.
        if clf_distil_scope not in (
            "old_classes", "replay_only", 
            "current_and_replay"
        ):
            raise ValueError(
                "clf_distil_scope must be 'old_classes', "
                "'replay_only', or 'current_and_replay'."
            )

        # Public single-teacher calls also honor a persistent Keras head's class identities.
        if not isinstance(teacher_labels, (tuple, list)) and \
        self._current_teacher_spec("classifier") is None and \
        self._uses_mapped_classifier_teachers():
            teacher_labels = tuple([teacher_labels])

        # Multiple teachers retain separate supports, argmax targets, and row normalizers.
        if isinstance(teacher_labels, (tuple, list)):
            return self._compute_dual_classifier_distillation(
                teacher_labels, 
                distil_classes, 
                clf_distil_type, 
                clf_distil_temperature, 
                clf_distil_scope, 
                clf_distil_loss_mask, 
                classes, 
                replay_mask, 
                student_logits, 
                kd_loss_allocator, 
                x0, 
                update_teacher_metrics, 
                teacher_loss_weight
            )
        return self._compute_single_classifier_distillation(
            teacher_labels, distil_classes, clf_distil_type, clf_distil_loss_mask, 
            classes, replay_mask, clf_distil_temperature, clf_distil_scope, 
            kd_loss_allocator, x0, student_logits, teacher_loss_weight
        )

    def _compute_single_classifier_distillation(
        self, 
        teacher_labels: tf.Tensor, 
        distil_classes: tf.Tensor, 
        clf_distil_type: str, 
        clf_distil_loss_mask: tf.Tensor | None = None, 
        classes: tf.Tensor | None = None, 
        replay_mask: tf.Tensor | None = None, 
        clf_distil_temperature: float = 1., 
        clf_distil_scope: str = "current_and_replay", 
        kd_loss_allocator: Callable | None = None, 
        x0: tf.Tensor | None = None, 
        student_logits: tf.Tensor | None = None, 
        teacher_loss_weight: float | None = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Reduce one validated CE/KL term after class columns have been aligned.

        Teacher support must occupy the leading student columns. The caller validates
        loss options and class mapping. Teacher scores are detached and cast with
        student scores to the policy variable dtype; shared teacher support is
        renormalized, excess columns truncated, and missing student columns padded
        with zeros. Hard KD uses argmax sparse CE; soft KD uses temperature-scaled KL
        with a squared-temperature factor. Masked reduction divides by selected weight
        with divide_no_nan. This helper does not update role metrics or model weights.

        Args:
            teacher_labels (tf.Tensor): Floating probabilities [B,Ct]; eligible rows must retain
                positive mass on shared columns.
            distil_classes (tf.Tensor): Floating student probabilities [B,Cs], returned unchanged
                after loss calculation.
            clf_distil_type (str): Validated hard or soft loss selector.
            clf_distil_loss_mask (tf.Tensor | None): Floating/bool per-row weights [B]; None
                (default) gives all rows unit weight before scope selection.
                Defaults to ``None``.
            classes (tf.Tensor | None): Sparse integer student class IDs [B]; defaults to None,
                permitted unless scope is old_classes.
            replay_mask (tf.Tensor | None): Boolean/numeric replay indicator [B]; defaults to None,
                permitted unless scope is replay_only.
            clf_distil_temperature (float): Validated positive temperature for soft KD; defaults to
                1.0, preserving teacher probabilities.
            clf_distil_scope (str): current_and_replay (default), old_classes or replay_only. The
                latter two intersect their selection with the explicit mask.
                Defaults to ``'current_and_replay'``.
            kd_loss_allocator (Callable | None): Optional callable (wrapper, row_losses, mask, x0,
                classes, replay_mask) returning a scalar reduced loss. Defaults to None for the
                selected-weight mean. It receives variable-dtype [B] losses and optional [B]
                weights.
            x0 (tf.Tensor | None): Floating clean model-coordinate images [B,H,W,C] passed unchanged
                to the allocator; None (default) supplies no image context.
                Defaults to ``None``.
            student_logits (tf.Tensor | None): Floating same-pass pre-softmax scores [B,Cs]. None
                (default) uses log of strictly positive student probabilities; required for
                saturated hard or soft KD.
                Defaults to ``None``.
            teacher_loss_weight (float | None): Scalar multiplier applied after reduction. None
                (default) uses previous_teacher_clf_loss_weight, falling back to 1.0 on objects
                without that attribute.
                Defaults to ``None``.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Scalar weighted loss in self.dtype_policy.variable_dtype
                and the unchanged student probability tensor [B,Cs] in its original dtype. A custom
                allocator owns its own reduction output contract.

        Raises:
            ValueError: old_classes lacks classes or replay_only lacks replay_mask.
            tf.errors.InvalidArgumentError: Teacher/student widths have no shared support, an
                eligible teacher row has zero retained mass, or probability-only KD contains
                nonpositive student probabilities. Allocator/cross-entropy errors propagate.
        """

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        stable_distil_classes = tf.cast(distil_classes, stable_dtype)
        teacher_labels = tf.stop_gradient(tf.cast(teacher_labels, stable_dtype))

        teacher_width = tf.shape(teacher_labels)[-1]
        student_width = tf.shape(distil_classes)[-1]
        tf.debugging.assert_positive(
            tf.minimum(teacher_width, student_width), 
            message="Teacher and student need shared class support."
        )
        known_teacher_labels = teacher_labels[:, :student_width]
        known_teacher_labels = tf.math.divide_no_nan(
            known_teacher_labels, 
            tf.reduce_sum(
                known_teacher_labels, 
                axis=-1, 
                keepdims=True
            )
        )
        missing_width = tf.maximum(
            student_width - tf.shape(known_teacher_labels)[-1], 0
        )
        teacher_labels = tf.pad(
            known_teacher_labels, 
            [[0, 0], [0, missing_width]]
        )

        # Convert teacher probabilities to labels for hard distillation.
        if clf_distil_type == "hard":
            hard_labels = tf.argmax(
                teacher_labels, 
                axis=-1, 
                output_type=tf.int32
            )
            # Direct probability callers remain valid while their scores are recoverable.
            if student_logits is None:
                tf.debugging.assert_positive(
                    stable_distil_classes, 
                    message="Saturated hard KD requires same-pass student_logits."
                )
                student_logits = tf.math.log(stable_distil_classes)
            clf_distil_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=hard_labels, 
                logits=tf.cast(student_logits, stable_dtype)
            )
        # Preserve the complete teacher distribution for soft distillation.
        else:
            # Preserve the original teacher target at unit temperature.
            if clf_distil_temperature == 1.:
                teacher_soft = teacher_labels
            # Soften only the known teacher vocabulary, then append exact new-class zeros.
            else:
                positive = known_teacher_labels > 0.
                teacher_log_probs = tf.where(
                    positive, 
                    tf.math.log(tf.where(positive, known_teacher_labels, tf.ones_like(known_teacher_labels))), 
                    tf.constant(-np.inf, dtype=stable_dtype)
                )
                # Ineligible zero-mass rows need finite placeholders; eligible rows are rejected below.
                teacher_log_probs = tf.where(
                    tf.reduce_any(positive, axis=-1, keepdims=True), 
                    teacher_log_probs, tf.zeros_like(teacher_log_probs)
                )
                teacher_soft = tf.nn.softmax(
                    teacher_log_probs
                    / tf.cast(tf.convert_to_tensor(clf_distil_temperature, dtype_hint=stable_dtype), stable_dtype), 
                    axis=-1
                )

                # New student classes are outside the frozen teacher support;
                # append exact zeros only after temperature softening.
                teacher_soft = tf.pad(
                    teacher_soft, 
                    [[0, 0], [0, missing_width]]
                )
            # Direct probability callers remain supported without clipping away gradients.
            if student_logits is None:
                tf.debugging.assert_positive(
                    stable_distil_classes, 
                    message="Saturated soft KD requires same-pass student_logits."
                )
                student_logits = tf.math.log(stable_distil_classes)

            student_log_probs = tf.nn.log_softmax(
                tf.cast(student_logits, stable_dtype)
                / tf.cast(tf.convert_to_tensor(clf_distil_temperature, dtype_hint=stable_dtype), stable_dtype), 
                axis=-1
            )
            clf_distil_loss = tf.reduce_sum(
                tf.math.xlogy(teacher_soft, teacher_soft)
                - teacher_soft * student_log_probs, 
                axis=-1
            ) * tf.cast(tf.convert_to_tensor(clf_distil_temperature ** 2, dtype_hint=stable_dtype), stable_dtype)

        clf_distil_loss = tf.cast(clf_distil_loss, stable_dtype)
        scope_mask = None
        # Select all examples whose ground-truth class exists in the teacher.
        if clf_distil_scope == "old_classes":
            # Require labels before deriving membership in the old vocabulary.
            if classes is None:
                raise ValueError(
                    "classes are required for clf_distil_scope='old_classes'."
                )

            scope_mask = tf.reshape(classes, tuple([-1])) < tf.cast(
                teacher_width, 
                classes.dtype
            )
        # Replay provenance cannot be inferred from class IDs in cumulative CL.
        elif clf_distil_scope == "replay_only":
            # Require the learner's explicit row-level provenance indicator.
            if replay_mask is None:
                raise ValueError(
                    "replay_mask is required for "
                    "clf_distil_scope='replay_only'."
                )

            scope_mask = tf.reshape(replay_mask, tuple([-1]))

        # Cast a supplied classifier mask before combining it with the KD scope.
        clf_distil_loss_mask = tf.cast(clf_distil_loss_mask, stable_dtype) \
                            if clf_distil_loss_mask is not None else None

        # Intersect the scope selector with any classifier-training mask.
        if scope_mask is not None:
            scope_mask = tf.cast(scope_mask, stable_dtype)
            # Use the KD scope alone or intersect it 
            # multiplicatively with existing classifier weights.
            clf_distil_loss_mask = scope_mask if clf_distil_loss_mask is None \
                                else clf_distil_loss_mask * scope_mask

        retained_mass = tf.reduce_sum(known_teacher_labels, axis=-1)
        eligible = tf.ones_like(retained_mass, dtype=tf.bool) if clf_distil_loss_mask is None \
                    else clf_distil_loss_mask > 0.
        tf.debugging.assert_positive(
            tf.where(eligible, retained_mass, tf.ones_like(retained_mass)), 
            message="Teacher probabilities need positive mass on shared classes for eligible rows."
        )

        weight = getattr(self, "previous_teacher_clf_loss_weight", 1.) \
                if teacher_loss_weight is None else teacher_loss_weight
        # A dedicated runtime reducer can change independent KD allocation only.
        if kd_loss_allocator is not None:
            return tf.cast(tf.convert_to_tensor(weight, dtype_hint=stable_dtype), stable_dtype) * kd_loss_allocator(
                self, 
                clf_distil_loss, 
                clf_distil_loss_mask, 
                x0, 
                classes, 
                replay_mask
            ), distil_classes

        # Normalize by selected exposure when masked; otherwise average the complete batch.
        clf_distil_loss = tf.math.divide_no_nan(
            tf.reduce_sum(
                clf_distil_loss * 
                clf_distil_loss_mask
            ), 
            tf.reduce_sum(clf_distil_loss_mask)
        ) if clf_distil_loss_mask is not None else tf.reduce_mean(clf_distil_loss)

        weight = getattr(self, "previous_teacher_clf_loss_weight", 1.) \
                if teacher_loss_weight is None else teacher_loss_weight

        return tf.cast(tf.convert_to_tensor(weight, dtype_hint=stable_dtype), stable_dtype) * clf_distil_loss, distil_classes

    def compute_clf_distil_ctr_loss(
        self, 
        classes: tf.Tensor, 
        classes_pred_list: list[tf.Tensor], 
        teacher_labels: tf.Tensor | None = None, 
        replay_mask: tf.Tensor | None = None, 
        loss_mask: tf.Tensor | None = None, 
        classes_logits_list: list[tf.Tensor | None] | None = None
    ) -> tuple[tf.Tensor | float, tf.Tensor | float]:
        """Compute ordinary or teacher-targeted token regularizer loss.

        Each non-None regularizer probability and optional logit tensor is floating
        [B,K]. Input sparse classes are integer [B]; loss_mask and replay provenance
        are Boolean/numeric [B]. Ordinary CE and KD use stable-policy arithmetic; output
        loss is rank zero and averaged probabilities are [B,K] in policy variable dtype.
        Base averaging supplies zero probabilities for an empty list; a supplied mask
        or active KD still evaluates its branch against that result. It does
        not update network or optimizer weights.

        Args:
            classes (tf.Tensor): Zero-based dataset targets shaped ``[B]``.
            classes_pred_list (list[tf.Tensor]): Available regularizer class
                probabilities.
            teacher_labels (tf.Tensor | None): Frozen teacher probabilities
                used by ``"distil"`` and ``"both"`` training modes.
                Defaults to ``None``.
                None is unused for ordinary token supervision but required to be replaced by valid
                floating [B,Ct] targets (or a mapped-role tuple) when token KD is active.
            replay_mask (tf.Tensor | None): Optional replay provenance passed
                to scoped teacher-targeted regularization.
                Defaults to ``None``.
            loss_mask (tf.Tensor | None): Per-row classifier weights [B]; None averages all
                ordinary token targets.
                None (default) uses the ordinary unmasked token loss; supplied bool/numeric [B] weights
                normalize the selected-row cross-entropy and KD.
                Defaults to ``None``.
            classes_logits_list (list[tf.Tensor | None] | None): Same-pass logits
                aligned with available regularizer probabilities, for stable CE and KD.
                Teacher-targeted token loss additionally uses the configured KD scope.
                Defaults to ``None``.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Scalar token loss and mean regularizer
            probabilities [B, student_width]. Normal mode uses ground-truth CE; distil
            mode replaces it with scoped KD; both averages the two losses. This helper
            does not gate disabled token loss: the surrounding aggregator does that.
            With no usable predictions, the base token helper supplies zero outputs.

        Raises:
            ValueError: The delegated distillation helper rejects missing scope metadata or
                unsupported KD settings. TensorFlow class-support/probability checks and
                cross-entropy errors propagate.
        """

        mixture_logits = None
        # Evaluate log(mean softmax(head)) without saturating any component probability.
        if self.use_clf_distil_ctr_loss and classes_logits_list is not None:
            stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
            # Absent regularizer depths contribute neither predictions nor mixture mass.
            log_probs = [tf.nn.log_softmax(tf.cast(value, stable_dtype), axis=-1)
                         for value in classes_logits_list if value is not None]
            # Form the mixture only when at least one regularizer head is present.
            if log_probs:
                mixture_logits = tf.reduce_logsumexp(tf.stack(log_probs), axis=0) \
                                - tf.math.log(tf.cast(len(log_probs), stable_dtype))

        clf_ctr_loss, clf_ctr_preds = self.compute_ctr_loss(
            classes, 
            classes_pred_list, 
            classes_logits_list=classes_logits_list, 
            loss_mask=loss_mask
        )

        # Retain ordinary regularizer targets when distillation is inactive.
        if not self.use_clf_distil_ctr_loss:
            return clf_ctr_loss, clf_ctr_preds

        regularizer_kwargs = getattr(
            self.network, 
            "clf_cls_token_regularizer_kwargs", 
            None
        )
        # Use generic regularizer settings only when no classifier-specific settings exist.
        regularizer_kwargs = getattr(
            self.network, 
            "cls_token_regularizer_kwargs", 
            {}
        ) if regularizer_kwargs is None else regularizer_kwargs
        regularizer_train_type = regularizer_kwargs.get(
            "train_type", 
            "normal"
        )
        # Replace or blend the regularizer target according to its train mode.
        if self.use_clf_ctr_loss and regularizer_train_type in (
            "distil", 
            "both"
        ):
            distil_ctr_loss, distil_ctr_preds = self.compute_clf_distil_loss(
                teacher_labels, 
                clf_ctr_preds, 
                regularizer_kwargs.get("distil_type", "hard"), 
                clf_distil_loss_mask=loss_mask, 
                classes=classes, 
                replay_mask=replay_mask, 
                student_logits=mixture_logits
            )
            # Distil-only regularizers replace ground truth; both mode averages true-label and
            # teacher losses.
            clf_ctr_loss = distil_ctr_loss if regularizer_train_type == "distil" \
                        else (clf_ctr_loss + distil_ctr_loss) / 2.

        return clf_ctr_loss, clf_ctr_preds

    def compute_clf_kl_ctr_distil_loss(
        self, 
        classes: tf.Tensor, 
        classes_pred_c: tf.Tensor | None, 
        clf_z_vals_list_c: list[tuple[tf.Tensor, tf.Tensor]] | None, 
        clf_regs_list_c: list[tf.Tensor | None] | None, 
        distil_classes_c: tf.Tensor | None = None, 
        classes_pred_u: tf.Tensor | None = None, 
        clf_z_vals_list_u: list[tuple[tf.Tensor, tf.Tensor]] | None = None, 
        clf_regs_list_u: list[tf.Tensor | None] | None = None, 
        distil_classes_u: tf.Tensor | None = None, 
        clf_loss_mask: tf.Tensor | None = None, 
        clf_train_type: TrainType | None = None, 
        kl_train_type: TrainType | None = None, 
        ctr_train_type: TrainType | None = None, 
        teacher_labels: tf.Tensor | None = None, 
        x0: tf.Tensor | None = None, 
        replay_mask: tf.Tensor | None = None, 
        logits_c: dict[str, object] | None = None, 
        logits_u: dict[str, object] | None = None, 
        training: bool | None = None
    ) -> tuple[
        tf.Tensor, tf.Tensor, tf.Tensor | float, tf.Tensor | float, 
        tf.Tensor | float, tf.Tensor, tf.Tensor | float, tf.Tensor | None
    ]:
        """Compute weighted classifier, classifier-KL, and token objectives.

        All class/regularizer probabilities and logits are real-floating [B,K]; each
        latent pair holds matching floating [B,...] mean/log-variance tensors. Loss
        outputs (the first five entries) are scalar policy-variable-dtype tensors,
        including zeros for disabled losses. Selected probabilities retain native or
        ensemble dtype; inactive token predictions are Python 0.0, inactive distillation
        predictions None. logits_c/logits_u=None each resolves to an empty dictionary.
        Active independent classifier-KD metrics may update; optimizer state does not.

        Args:
            classes (tf.Tensor): Zero-based targets ``[B]``.
            classes_pred_c (tf.Tensor | None): Conditional class probabilities
                ``[B,num_classes]``.
            clf_z_vals_list_c (list[tuple[tf.Tensor, tf.Tensor]] | None): Ordered
                conditional classifier latent mean/log-variance pairs.
            clf_regs_list_c (list[tf.Tensor | None] | None): Conditional
                classifier token predictions.
            distil_classes_c (tf.Tensor | None): Conditional distillation-head
                probabilities; uses primary probabilities when no token exists.
                Defaults to ``None``.
            classes_pred_u (tf.Tensor | None): Unconditional probabilities.
                Defaults to ``None``.
            clf_z_vals_list_u (list[tuple[tf.Tensor, tf.Tensor]] | None): Ordered
                unconditional classifier latent pairs.
                Defaults to ``None``.
            clf_regs_list_u (list[tf.Tensor | None] | None): Unconditional token
                predictions.
                Defaults to ``None``.
            distil_classes_u (tf.Tensor | None): Unconditional distillation-head
                probabilities; uses primary probabilities when no token exists.
                Defaults to ``None``.
            clf_loss_mask (tf.Tensor | None): Float per-example mask ``[B]``;
                None averages all cross-entropies.
                Defaults to ``None``.
            clf_train_type (TrainType | None): Probability source; None uses the
                configured ``clf_train_type``.
                Defaults to ``None``.
            kl_train_type (TrainType | None): Classifier KL source; None uses the
                base setting.
                Defaults to ``None``.
            ctr_train_type (TrainType | None): Classifier token-loss branch; None inherits
                self.ctr_train_type.
                Defaults to ``None``.
            teacher_labels (tf.Tensor | None): Frozen teacher probabilities required by
                active classifier KD
                or token regularizers in distil/both mode; ignored without those losses.
                Defaults to ``None``.
            x0 (tf.Tensor | None): Clean images required by ensemble loss.
                Defaults to ``None``.
            replay_mask (tf.Tensor | None): Optional replay provenance for
                scoped distillation losses.
                Defaults to ``None``.
            logits_c (dict[str, object] | None): Same-pass conditional classifier
                logits metadata; None supports direct probability-only helper calls.
                Defaults to None, resolved to an empty dict. Floating class_logits/distil_logits tensors
                are [B,K]; clf_regs_logits_list contains optional [B,K] tensors by depth.
            logits_u (dict[str, object] | None): Corresponding null-label metadata.
                Defaults to None, resolved to an empty dict with the same optional floating logits
                structure as logits_c.
            training (bool | None): Mode passed to the ensemble predictor.
                Defaults to ``None``.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor | float, tf.Tensor | float,
            tf.Tensor | float, tf.Tensor, tf.Tensor | float, tf.Tensor | None]:
            Weighted classifier total, raw classifier loss, classifier KL loss,
            classifier token loss, distillation loss, selected probabilities,
            averaged token probabilities, and distillation probabilities.

        Raises:
            ValueError: Enabled classifier distillation receives invalid scope/teacher metadata or
                incompatible role targets. Errors from active classifier/KL/token loss helpers
                propagate; disabled branches do not evaluate their losses.
        """

        clf_train_type = self.clf_train_type if clf_train_type is None else clf_train_type
        kl_train_type = self.kl_train_type if kl_train_type is None else kl_train_type
        ctr_train_type = self.ctr_train_type if ctr_train_type is None else ctr_train_type
        logits_c = {} if logits_c is None else logits_c
        logits_u = {} if logits_u is None else logits_u

        # Keep KD on the original head probabilities even when CE uses an ensemble.
        if getattr(self.network, "distil_token", None) is None:
            distil_classes_c, distil_classes_u = classes_pred_c, classes_pred_u

        # Select primary probabilities from the conditional or explicit null branch.
        clf_loss, classes_pred = self.compute_clf_loss(
            classes, 
            classes_pred=classes_pred_c if clf_train_type == "cond" else classes_pred_u, 
            clf_loss_mask=clf_loss_mask, 
            x0=x0, 
            training=training
        )
        # Compute classifier KL only when enabled, using its selected conditional/null statistics.
        # Select the latent statistics from the configured classifier KL branch.
        clf_kl_loss = VariationalAutoencoder.compute_kl(
            clf_z_vals_list_c if kl_train_type == "cond" else clf_z_vals_list_u, 
            sample_weight=clf_loss_mask, 
            dtype=self.dtype_policy.variable_dtype
        ) if self.use_clf_kl_loss else 0.
        # Compute classifier token loss only when enabled, using its selected branch predictions.
        # Select conditional or null-branch classifier regularizers for the token objective.
        clf_ctr_loss, clf_ctr_preds = self.compute_clf_distil_ctr_loss(
            classes, 
            classes_pred_list=clf_regs_list_c if ctr_train_type == "cond" else clf_regs_list_u, 
            teacher_labels=teacher_labels, 
            replay_mask=replay_mask, 
            loss_mask=clf_loss_mask, 
            # Use the same conditional/null branch as the auxiliary probability mixture.
            classes_logits_list=(logits_c if ctr_train_type == "cond" else logits_u).get(
                "clf_regs_logits_list"
            )
        ) if self.use_clf_ctr_loss else (0., 0.)
        # Compute classifier KD only when an active teacher objective exists. Distil the same
        # conditional or null branch used by primary classification.
        clf_distil_loss, distil_classes = self.compute_clf_distil_loss(
            teacher_labels, 
            distil_classes_c if clf_train_type == "cond" 
            else distil_classes_u, 
            clf_distil_loss_mask=clf_loss_mask, 
            classes=classes, 
            replay_mask=replay_mask, 
            kd_loss_allocator=getattr(
                self, 
                "_classifier_kd_allocator", 
                None
            ) if training is True else None, 
            x0=x0, 
            student_logits=(logits_c if clf_train_type == "cond" else logits_u).get(
                "distil_logits" if getattr(self.network, "distil_token", None) is not None
                else "class_logits"
            ), 
            update_teacher_metrics=True
        ) if self.use_clf_distil_loss else (0., None)

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        clf_loss = tf.cast(clf_loss, stable_dtype)
        clf_kl_loss = tf.cast(clf_kl_loss, stable_dtype)
        clf_ctr_loss = tf.cast(clf_ctr_loss, stable_dtype)
        clf_distil_loss = tf.cast(clf_distil_loss, stable_dtype)
        loss = (
            clf_loss * self.clf_loss_coef + 
            clf_kl_loss * self.kl_loss_coef + 
            clf_ctr_loss * self.ctr_loss_coef + 
            clf_distil_loss * self.clf_distil_loss_coef
        )

        outputs = (
            loss, clf_loss, clf_kl_loss, 
            clf_ctr_loss, clf_distil_loss, 
            classes_pred, clf_ctr_preds, 
            distil_classes
        )

        return outputs

    def compute_batch_diffusion_losses(
        self, 
        diffusion_mask: tf.Tensor | None, 
        **loss_inputs: object
    ) -> tuple[tf.Tensor | None, ...]:
        """Compute diffusion objectives only on their allocated batch rows.

        diffusion_mask is bool [B]. Selected tensor leaves retain dtype and replace the
        leading batch size with M selected rows. With a selector, all tensor outputs
        are cast to policy variable dtype; disabled split-noise positions are restored
        to None. With selector None, the original aggregator output types/sentinels are
        returned unchanged. No optimizer update is performed.

        Args:
            diffusion_mask (tf.Tensor | None): Boolean diffusion allocation,
                or None to preserve the ordinary full-batch loss path.
            **loss_inputs (object): Arguments for the inherited diffusion loss
                aggregator. Tensor leaves all have their batch dimension first;
                nested latent pairs, regularizers, and optional teacher targets
                retain their structure while their rows are selected.

        Returns:
            tuple[tf.Tensor | None, ...]: The inherited nine loss outputs. An
            empty diffusion allocation returns finite zero losses and empty
            token predictions without evaluating empty-batch means.

        Raises:
            No extra validation is added. tf.boolean_mask reports incompatible row selectors;
                selected diffusion-loss errors propagate from the inherited aggregator. The empty
                branch avoids empty-batch means.
        """

        # Zero-fraction training retains the original aggregation and return structure.
        if diffusion_mask is None:
            return self.compute_noise_distil_image_kl_ctr_loss(**loss_inputs)

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        zero = tf.constant(0., dtype=stable_dtype)


        def select_rows(value: object) -> object:
            """Select batch rows from tensors while preserving optional metadata.

            Args:
                value (object): Tensor leaf with leading batch dimension, or optional
                    non-tensor loss metadata from the captured nested argument mapping.

            Returns:
                selected (object): Same-dtype tensor with eligible rows retained, or
                    the unchanged non-tensor leaf.

            Raises:
                No explicit exceptions are raised. tf.boolean_mask reports a selector incompatible with
                    the tensor's leading batch dimension.
            """

            # Keep same-pass pre-softmax scores aligned with selected token probabilities.
            if tf.is_tensor(value):
                selected = tf.boolean_mask(value, diffusion_mask)
                logits = getattr(value, "_keras_logits", None)
                # Only probability leaves carry this same-pass score metadata.
                if logits is not None:
                    selected._keras_logits = tf.boolean_mask(logits, diffusion_mask)
                return selected
            return value


        selected_inputs = tf.nest.map_structure(select_rows, loss_inputs)


        def compute_selected() -> tuple[tf.Tensor, ...]:
            """Evaluate diffusion losses only for the selected nonempty rows.

            Uses captured selected_inputs; disabled None diagnostics are temporarily
            replaced with finite scalar zeros so both tf.cond branches have matching
            tensor structures. Role metric updates performed by the aggregator are retained.

            Args:
                None.

            Returns:
                tuple[tf.Tensor, ...]: Nine outputs in the inherited diffusion-loss order, cast to
                    policy variable dtype. Losses are scalars; enabled token predictions are
                    [M,num_classes] for M selected rows.

            Raises:
                Errors from the inherited enabled diffusion objectives and casting their returned values
                    propagate; no independent exceptions are raised here.
            """

            outputs = self.compute_noise_distil_image_kl_ctr_loss(**selected_inputs)

            return tuple(
                tf.cast(value, stable_dtype) 
                if value is not None else zero
                for value in outputs
            )


        def empty_selected() -> tuple[tf.Tensor, ...]:
            """Construct finite placeholders for an empty diffusion allocation.

            Does not call the network, evaluate means on empty tensors, or update metrics.

            Args:
                None.

            Returns:
                tuple[tf.Tensor, ...]: Nine policy-variable-dtype tensors. Eight losses are scalar zero;
                    the last is zero predictions [0,num_classes] when token loss is enabled, otherwise
                    scalar zero.

            Raises:
                This finite-placeholder constructor raises no explicit exceptions.
            """

            token_predictions = tf.zeros(
                (0, self.network.num_classes), 
                dtype=stable_dtype
            ) if self.use_ctr_loss else zero

            return (zero, zero, zero, zero, zero, zero, zero, zero, token_predictions)


        outputs = tf.cond(
            tf.reduce_any(diffusion_mask), 
            compute_selected, 
            empty_selected
        )
        # Disabled split diagnostics remain None for the existing reporting API.
        if not self.show_separate_noise_losses:
            outputs = (*outputs[:2], None, None, *outputs[4:])

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
        distil_classes: tf.Tensor | None = None, 
        use_total_loss: bool | None = None, 
        use_kl_loss: bool | None = None, 
        use_ctr_loss: bool | None = None, 
        use_clf_distil_loss: bool | None = None, 
        clf_distil_acc_mask: tf.Tensor | None = None, 
        clf_ctr_mask: tf.Tensor | None = None
    ) -> dict[str, tf.Tensor]:
        """Update classifier metric trackers and return current values.

        Loss means weight batches by their represented row counts; accuracies use
        selected examples directly. Total accuracy combines available heads on the
        classifier mask, with auxiliary components zeroed outside their own valid
        scope. Probability sums need not equal one because categorical accuracy uses
        argmax. Dynamic singleton heads are padded only for TensorFlow 2.10 metric
        compatibility, leaving the actual network output vocabulary unchanged.

        Loss arguments are real-floating scalars. Class/probability tensors are integer
        [B] and floating [B,K]; accuracy/KD/token masks are bool or numeric [B]. Returned
        values are running scalar tensors in each tracker's policy variable dtype.
        This method updates only metric accumulators, leaving model/optimizer weights
        and random streams unchanged.

        Args:
            clf_loss (tf.Tensor): Required scalar classifier loss.
            classes (tf.Tensor): Zero-based targets ``[B]``.
            classes_pred (tf.Tensor): Class probabilities ``[B,num_classes]``.
            clf_acc_mask (tf.Tensor | None): Boolean selector ``[B]`` for
                masked loss and accuracy tracking; None selects all samples.
                Defaults to ``None``.
            total_loss (tf.Tensor | None): Required when total tracking is on.
                Defaults to ``None``.
            clf_kl_loss (tf.Tensor | None): Required when classifier KL is on.
                Defaults to ``None``.
            clf_ctr_loss (tf.Tensor | None): Required when token loss is on.
                Defaults to ``None``.
            clf_distil_loss (tf.Tensor | None): Required when distillation is
                active.
                Defaults to ``None``.
            clf_ctr_preds (tf.Tensor | None): Token probabilities for accuracy.
                Defaults to ``None``.
            distil_classes (tf.Tensor | None): Distillation-head class
                probabilities.
                Defaults to ``None``.
            use_total_loss (bool | None): Explicit total tracker switch; None enables it
                when classifier KL,
                token regularization, or independent-head distillation is enabled.
                Defaults to ``None``.
            use_kl_loss (bool | None): Tracker override; None follows self.use_clf_kl_loss.
                Defaults to ``None``.
            use_ctr_loss (bool | None): Tracker override; None follows
                self.use_clf_ctr_loss.
                Defaults to ``None``.
            use_clf_distil_loss (bool | None): Tracker override; None follows
                self.use_clf_distil_loss.
                Defaults to ``None``.
            clf_distil_acc_mask (tf.Tensor | None): Optional boolean selector for
                distillation loss/accuracy accounting. ``None`` preserves the
                historical classifier-mask scope.
                Defaults to ``None``.
            clf_ctr_mask (tf.Tensor | None): Rows contributing to classifier
                token loss and accuracy. ``None`` uses ``clf_acc_mask``.
                Defaults to ``None``.

        Returns:
            dict[str, tf.Tensor]: Current classifier metrics keyed by name.

        Raises:
            AssertionError: If a requested optional metric lacks its input.
        """

        # Keras accuracy expects tensors with ndim; raw tf.Variable inputs lack that attribute.
        classes = tf.convert_to_tensor(classes)
        classes_pred = tf.convert_to_tensor(classes_pred)
        clf_ctr_preds = None if clf_ctr_preds is None else tf.convert_to_tensor(clf_ctr_preds)
        distil_classes = None if distil_classes is None else tf.convert_to_tensor(distil_classes)

        clf_acc_mask = tf.ones_like(classes, dtype=tf.bool) \
                    if clf_acc_mask is None else tf.cast(clf_acc_mask, tf.bool)
        clf_distil_acc_mask = clf_acc_mask if clf_distil_acc_mask is None \
                            else tf.cast(clf_distil_acc_mask, tf.bool)
        clf_ctr_mask = clf_acc_mask if clf_ctr_mask is None \
                        else tf.cast(clf_ctr_mask, tf.bool)
        use_kl_loss = self.use_clf_kl_loss if use_kl_loss is None else use_kl_loss
        use_ctr_loss = self.use_clf_ctr_loss if use_ctr_loss is None else use_ctr_loss
        use_clf_distil_loss = self.use_clf_distil_loss if use_clf_distil_loss is None \
                            else use_clf_distil_loss
        use_total_loss = use_kl_loss or use_ctr_loss or use_clf_distil_loss \
                        if use_total_loss is None else use_total_loss

        stable_dtype = tf.as_dtype(self.dtype_policy.variable_dtype)
        batch_weight = tf.cast(tf.shape(classes)[0], stable_dtype)
        selected_weight = tf.reduce_sum(
            tf.cast(clf_acc_mask, stable_dtype)
        )
        clf_distil_selected_weight = tf.reduce_sum(
            tf.cast(clf_distil_acc_mask, stable_dtype)
        )
        clf_ctr_selected_weight = tf.reduce_sum(
            tf.cast(clf_ctr_mask, stable_dtype)
        )

        # TensorFlow 2.10 misreads one-column predictions as binary outputs.
        if self.network.dynamic_num_classes:
            # Pad the primary prediction while the dynamic head has one class.
            if classes_pred.shape[-1] == 1:
                classes_pred = tf.concat([
                    classes_pred, 
                    tf.zeros_like(classes_pred)
                ], axis=-1)

            # Apply the same compatibility padding to token predictions.
            if use_ctr_loss and tf.is_tensor(clf_ctr_preds) and clf_ctr_preds.shape[-1] == 1:
                clf_ctr_preds = tf.concat([
                    clf_ctr_preds, 
                    tf.zeros_like(clf_ctr_preds)
                ], axis=-1)

            # Apply the same compatibility padding to distillation predictions.
            if tf.is_tensor(distil_classes) and distil_classes.shape[-1] == 1:
                distil_classes = tf.concat([
                    distil_classes, 
                    tf.zeros_like(distil_classes)
                ], axis=-1)

        self.clf_loss_tracker.update_state(
            clf_loss, 
            sample_weight=selected_weight
        )
        # Static batch shapes keep XLA metric counts correct for partial allocations.
        self.accuracy_tracker.update_state(
            classes, 
            classes_pred, 
            sample_weight=tf.cast(clf_acc_mask, stable_dtype)
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
                sample_weight=selected_weight
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
                sample_weight=clf_ctr_selected_weight
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
                clf_ctr_preds, 
                sample_weight=tf.cast(clf_ctr_mask, stable_dtype)
            )
            results.update({
                self.clf_ctr_accuracy_tracker.name: 
                self.clf_ctr_accuracy_tracker.result()
            })

        # Use one deployable mixture; target/replay scope masks only select diagnostics.
        if (use_ctr_loss and self.ctr_acc_coef > 0.) or (
            use_clf_distil_loss and self.clf_distil_acc_coef > 0.
        ):
            total_preds = classes_pred * self.clf_acc_coef

            # Add classifier-regularizer predictions when weighted in.
            if use_ctr_loss and self.ctr_acc_coef > 0.:
                require(
                    clf_ctr_preds is not None, 
                    "ctr_acc_coef > 0 requires clf_ctr_preds."
                )

                ctr_component = clf_ctr_preds
                total_preds += ctr_component * self.ctr_acc_coef

            # Add the selected distillation predictions when weighted in.
            if use_clf_distil_loss and self.clf_distil_acc_coef > 0.:
                require(
                    distil_classes is not None, 
                    "clf_distil_acc_coef > 0 requires distil_classes."
                )

                distil_component = distil_classes

                total_preds += distil_component * self.clf_distil_acc_coef

            self.total_accuracy_tracker.update_state(
                classes, 
                total_preds, 
                sample_weight=tf.cast(clf_acc_mask, stable_dtype)
            )
            results.update({
                self.total_accuracy_tracker.name: 
                self.total_accuracy_tracker.result()
            })

        # Track the distillation objective and prediction accuracy.
        if use_clf_distil_loss:
            require(
                clf_distil_loss is not None and distil_classes is not None, 
                "When use_clf_distil_loss is True, clf_distil_loss "
                "and distil_classes cannot be None."
            )

            self.clf_distil_loss_tracker.update_state(
                clf_distil_loss, 
                sample_weight=clf_distil_selected_weight
            )
            self.clf_distil_acc_tracker.update_state(
                classes, 
                distil_classes, 
                sample_weight=tf.cast(clf_distil_acc_mask, stable_dtype)
            )

            results.update({
                self.clf_distil_loss_tracker.name: 
                self.clf_distil_loss_tracker.result(), 
                self.clf_distil_acc_tracker.name: 
                self.clf_distil_acc_tracker.result()
            })

        # The dual total is the weighted sum of independently normalized teacher populations.
        if use_clf_distil_loss and self._uses_mapped_classifier_teachers():
            combined = tf.zeros((), dtype=stable_dtype)
            for spec in self._classifier_teacher_specs():
                tracker = getattr(self, spec["role"] + "_teacher_clf_loss_tracker")
                results[tracker.name] = tracker.result()
                combined += tf.cast(
                    tf.convert_to_tensor(spec["weight"], 
                    dtype_hint=stable_dtype
                ), stable_dtype) * tracker.result()

            results[self.clf_distil_loss_tracker.name] = combined

        return results


def run_self_tests() -> dict[str, str]:
    """Run CPU-small joint diffusion/classification wrapper tests.

    Creates real tiny networks and runs eager training/evaluation on synthetic
    float32 image batches; no dataset download is needed. Clears the global Keras
    session before starting, sets TensorFlow's global seed to 106, and clears
    the session again on successful completion. It does not restore the previous
    random state. Assertion or TensorFlow failures stop the remaining checks.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"DiffusionClassifier": "passed"}`` after masking,
        conditional/unconditional, ensemble, auxiliary-loss, metric,
        optimization, evaluation, progressive-growth, and rejection checks.

    Raises:
        AssertionError: A numerical/state assertion fails or an invalid-input probe
                unexpectedly succeeds. TensorFlow execution errors propagate unchanged.
    """

    tf.keras.backend.clear_session()
    tf.random.set_seed(106)


    from diffusion.models.transformer.di_t_classifier import DiTClassifier


    def make_network(**overrides: object) -> DiTClassifier:
        """Construct a fresh tiny classifier network with safe mutable IDs.

        Creates new configuration containers on every call. Overrides replace their
        corresponding defaults as a whole, so nested mappings are not deep-merged.
        The default build uses two classes, CFG, four diffusion steps, 4x4 one-channel
        images, 2x2 patches, embedding width four, one attention head, and MLP ratio one.

        The default encoder has one block; the classifier uses one attention head
        and MLP ratio one. Fresh feature_aggregation_ids_dict={1: (-1,)} and
        clf_connection_ids_dict={-1: (-1,)} isolate mutable routing configuration
        between fixtures. Remaining DiTClassifier options keep their own defaults.

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
            "feature_aggregation_ids_dict": {1: tuple([-1])}, 
            "clf_connection_ids_dict": {-1: tuple([-1])}, 
            **overrides
        }

        return DiTClassifier(**config)


    def make_wrapper(**overrides: object) -> DiffusionClassifier:
        """Construct and compile a fresh tiny classifier wrapper.

        Uses a supplied network override when present, otherwise the tiny factory
        network. A factory network is eagerly constructed while resolving that
        override even when a supplied network wins. The wrapper defaults to EMA
        evaluation, a linear schedule, two sampling steps, and eager Adam(1e-3)
        optimization with mean squared error. Overrides replace wrapper options
        before construction; compilation options are fixed by this fixture.

        Sets wrapper seed 37 and p_uncond=1.0 so every training conditioning label
        is dropped unless the caller supplies a different dropout probability.

        Args:
            **overrides (object): Wrapper arguments replacing test defaults.

        Returns:
            DiffusionClassifier: An eagerly compiled wrapper.
        """

        network = overrides.pop("network", make_network())
        config = {
            "network": network, 
            "preprocess_type": None, 
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
            run_eagerly=True 
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
        "classifier_loss", "clf_kl_loss", "clf_ctr_loss", "clf_distil_loss", 
        "total_accuracy", "classifier_accuracy", "clf_ctr_accuracy", 
        "clf_distil_acc"
    ]

    threshold_only = make_wrapper(
        mask_by_nulls=False, 
        mask_by_t_threshold=True, 
        mask_t_percentage=25, 
        p_uncond=0.0 
    )
    assert threshold_only.mask_by_nulls is False
    assert threshold_only.mask_by_t_threshold is True
    assert int(threshold_only.filter_t_threshold) == 0
    empty_threshold = make_wrapper(
        mask_by_nulls=False, 
        mask_t_percentage=0, 
        p_uncond=0.0 
    )
    full_threshold = make_wrapper(
        mask_by_nulls=False, 
        mask_t_percentage=100, 
        p_uncond=0.0 
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
        network_name="raw", training=False
    )
    assert len(conditional_only) == 6
    assert conditional_only[0][0].shape == images.shape
    assert conditional_only[0][1] is None
    assert conditional_only[3][0].shape == (2, 2)
    both = wrapper.call_network(
        x_t, t, cond_labels, null_labels, scale=2.0, 
        network_name="raw", training=False
    )
    assert all(pair[1] is not None for pair in both)

    mask = tf.constant([1.0, 0.0])
    clf_values = wrapper.compute_clf_kl_ctr_distil_loss(
        classes, 
        both[3][0], both[5][0], both[4][0], 
        classes_pred_u=both[3][1], 
        clf_z_vals_list_u=both[5][1], 
        clf_regs_list_u=both[4][1], 
        clf_loss_mask=mask, 
        clf_train_type="cond" 
    )
    assert len(clf_values) == 8 and float(clf_values[1]) >= 0.0
    empty_values = wrapper.compute_clf_kl_ctr_distil_loss(
        classes, 
        both[3][0], both[5][0], both[4][0], 
        classes_pred_u=both[3][1], 
        clf_z_vals_list_u=both[5][1], 
        clf_regs_list_u=both[4][1], 
        clf_loss_mask=tf.zeros(tuple([2])), 
        clf_train_type="uncond"
    )
    assert float(empty_values[1]) == 0.0

    # KL/token epoch means and token accuracy must follow their row masks.
    wrapper.get_clf_results_dict(
        tf.constant(0.), 
        classes, 
        tf.constant([[1., 0.], [0., 1.]]), 
        clf_acc_mask=tf.constant([True, False]), 
        clf_kl_loss=tf.constant(0.), 
        clf_ctr_loss=tf.constant(0.), 
        clf_ctr_preds=tf.constant([[1., 0.], [1., 0.]]), 
        clf_ctr_mask=tf.constant([True, False]), 
        use_total_loss=False, 
        use_kl_loss=True, 
        use_ctr_loss=True, 
        use_clf_distil_loss=False
    )
    masked_token_metrics = wrapper.get_clf_results_dict(
        tf.constant(0.), 
        classes, 
        tf.constant([[1., 0.], [0., 1.]]), 
        clf_acc_mask=tf.constant([True, True]), 
        clf_kl_loss=tf.constant(1.), 
        clf_ctr_loss=tf.constant(1.), 
        clf_ctr_preds=tf.constant([[1., 0.], [0., 1.]]), 
        clf_ctr_mask=tf.constant([True, True]), 
        use_total_loss=False, 
        use_kl_loss=True, 
        use_ctr_loss=True, 
        use_clf_distil_loss=False
    )
    tf.debugging.assert_near(
        masked_token_metrics[wrapper.clf_ctr_accuracy_tracker.name], 
        1.
    )
    tf.debugging.assert_near(
        masked_token_metrics[wrapper.clf_kl_loss_tracker.name], 
        2. / 3.
    )
    tf.debugging.assert_near(
        masked_token_metrics[wrapper.clf_ctr_loss_tracker.name], 
        2. / 3.
    )
    for metric in wrapper.metrics:
        metric.reset_state()

    scoped_ctr = make_wrapper(clf_acc_coef=0., ctr_acc_coef=1.)
    scoped_ctr_results = scoped_ctr.get_clf_results_dict(
        tf.constant(0.), 
        classes, 
        tf.constant([[1., 0.], [1., 0.]]), 
        clf_ctr_loss=tf.constant(0.), 
        clf_ctr_preds=tf.constant([[1., 0.], [0., 1.]]), 
        clf_ctr_mask=tf.constant([True, False]), 
        use_total_loss=False, 
        use_kl_loss=False, 
        use_ctr_loss=True, 
        use_clf_distil_loss=False
    )
    tf.debugging.assert_near(
        scoped_ctr_results[scoped_ctr.total_accuracy_tracker.name], 
        # Scope selects the head diagnostic; both label-free auxiliary predictions are correct.
        1.
    )

    train_results = wrapper.train_step((images, classes))
    assert {
        "loss", "noise_loss", "classifier_loss", "classifier_accuracy"
    } <= set(train_results)
    test_results = wrapper.test_step((images, classes))
    assert {
        "loss", "noise_loss", "image_loss", "classifier_loss", 
        "classifier_accuracy"
    } <= set(test_results)
    separate_noise_wrapper = make_wrapper(
        mask_by_nulls=True, 
        show_separate_noise_losses=True
    )
    separate_noise_results = separate_noise_wrapper.train_step(
        (images, classes)
    )
    assert {
        "total_noise_loss", "cond_noise_loss", "uncond_noise_loss", 
        "classifier_loss"
    } <= set(separate_noise_results)
    assert "noise_loss" not in separate_noise_results
    dataset = tf.data.Dataset.from_tensor_slices((images, classes)).batch(2)
    history = wrapper.fit(dataset, epochs=1, verbose=0)
    assert len(history.history["classifier_loss"]) == 1
    evaluation = wrapper.evaluate(
        dataset, network_name="raw", return_dict=True, verbose=0
    )
    assert "classifier_accuracy" in evaluation

    try:
        DiffusionClassifier(
            network=make_network(clf_distil_token_type="new_weight"), 
            clf_distil_loss_coef=1.0, 
            mask_by_nulls=False, 
            p_uncond=0.0, 
            use_ema=False, 
            test_network_name="raw", 
            test_steps=2
        )
    except AssertionError:
        pass
    # An enabled token-distillation objective must reject a missing required teacher.
    else:
        raise AssertionError(
            "A configured token objective requires a teacher by default."
        )

    positional_teacher = make_network()
    positional_compatibility = DiffusionClassifier(
        teacher_network=positional_teacher, 
        clf_distil_type="soft", 
        mask_by_nulls=False, 
        network=make_network(), 
        use_ema=False, 
        test_network_name="raw", 
        test_steps=2
    )
    assert positional_compatibility.teacher_network is positional_teacher
    assert positional_compatibility.clf_distil_type == "soft"
    assert positional_compatibility.mask_by_nulls is False
    assert positional_compatibility.defer_teacher is True

    mismatched_teacher_wrapper = make_wrapper(
        scheduler_name="quadratic", 
        mask_by_nulls=False, 
        p_uncond=0.0
    )
    try:
        DiffusionClassifier(
            teacher_network=mismatched_teacher_wrapper, 
            network=make_network(clf_distil_token_type="new_weight"), 
            clf_distil_loss_coef=1., 
            mask_by_nulls=False, 
            p_uncond=0.0, 
            use_ema=False, 
            test_network_name="raw", 
            scheduler_name="linear", 
            test_steps=2
        )
    except ValueError as error:
        assert "scheduler_name must match" in str(error)
    # Classifier teachers must not discard incompatible wrapper schedule metadata.
    else:
        raise AssertionError(
            "A classifier-only wrapper teacher must keep schedule metadata."
        )

    continual_student = make_wrapper(
        network=make_network(
            num_classes=None, 
            clf_distil_token_type="new_weight"
        ), 
        defer_teacher=True, 
        noise_distil_loss_coef=1.0, 
        clf_distil_loss_coef=1.0, 
        mask_by_nulls=False, 
        p_uncond=0.0
    )
    assert continual_student.teacher_network is None
    assert continual_student.use_clf_distil_loss is False
    assert continual_student.map_preprocess is False
    assert continual_student.accuracy_tracker.name == "cls_token_accuracy"
    assert continual_student.get_config()["defer_teacher"] is True
    assert "teacher_network" not in continual_student.get_config()

    try:
        continual_student.set_teacher_network(tf.keras.Sequential())
    except ValueError:
        pass
    # A single-output image teacher cannot supervise both class and noise objectives.
    else:
        raise AssertionError(
            "Combined noise and class distillation requires a compatible multi-output teacher."
        )
    external_teacher = make_network()
    continual_student.set_teacher_network(external_teacher)
    tf.debugging.assert_equal(
        continual_student._mask_unknown_teacher_labels(classes), 
        classes
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
    snapshot_times = tf.zeros(tuple([2]), dtype=tf.int32)
    snapshot_labels = tf.constant([1, 2], dtype=tf.uint8)
    source_outputs = continual_student.ema_network(
        (snapshot_images, snapshot_times, snapshot_labels), 
        training=False
    )
    teacher_outputs = teacher(
        (snapshot_images, snapshot_times, snapshot_labels), 
        training=False
    )
    for output_name in ("noises", "classes", "distil_classes"):
        tf.debugging.assert_near(
            source_outputs[output_name], 
            teacher_outputs[output_name]
        )

    continual_student._check_new_labels(
        y=tf.constant([2], dtype=tf.uint8), 
        verbose=False
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
        tf.constant([1, 0], dtype=tf.uint8)
    )
    assert continual_student.use_clf_distil_loss
    assert continual_student.use_classifier_distil
    assert continual_student.accuracy_tracker.name == "cls_token_accuracy"
    assert continual_student.map_preprocess
    assert continual_student.train_function is None
    assert continual_student.test_function is None

    new_task_images = images[:1]
    new_task_classes = tf.constant([2], dtype=tf.uint8)
    prepared_distillation = continual_student.prep_inputs_map(
        new_task_images, 
        new_task_classes
    )
    assert len(prepared_distillation) == 10
    assert prepared_distillation[-1].shape == (1, 2)
    distil_step = continual_student.train_step(prepared_distillation)
    assert {
        "noise_distil_loss", "clf_distil_loss", "clf_distil_acc"
    } <= set(distil_step)

    new_task_dataset = tf.data.Dataset.from_tensor_slices((
        new_task_images, 
        new_task_classes
    )).batch(1)
    distil_progressive = continual_student.fit_progressively(
        stage_tasks=[{"resolution": 8}], 
        x=new_task_dataset, 
        validation_data=new_task_dataset, 
        stages_verbose=False, 
        stage_epochs=1, 
        final_epochs=0, 
        verbose=0
    )
    assert len(distil_progressive.progressive_stages) == 1
    continual_student.set_teacher_network(None)
    assert continual_student.teacher_network is None
    assert continual_student.use_clf_distil_loss is False
    assert continual_student.map_preprocess is False

    from diffusion.models.transformer.di_t_encoder_decoder_classifier import (
        DiTEncoderDecoderClassifier
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
            "feature_aggregation_ids_dict": {1: tuple([-1])}, 
            "clf_connection_ids_dict": {-1: tuple([-1])}, 
            "clf_distil_token_type": "new_weight"
        }, 
        decoder_kwargs={
            "depth": 0, 
            "shift_inputs": False, 
            "use_unpatchify": True
        }
    )
    composite_student = make_wrapper(
        network=composite_network, 
        defer_teacher=True, 
        clf_distil_loss_coef=1.0, 
        mask_by_nulls=False, 
        p_uncond=0.0, 
        use_ema=False, 
        test_network_name="raw"
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
        tf.zeros(tuple([2]), dtype=tf.int32), 
        tf.constant([1, 2], dtype=tf.uint8)
    )
    composite_source_outputs = composite_student.network(
        composite_inputs, 
        training=False
    )
    composite_teacher_outputs = composite_teacher(
        composite_inputs, 
        training=False
    )
    for output_name in ("noises", "classes", "distil_classes"):
        tf.debugging.assert_near(
            composite_source_outputs[output_name], 
            composite_teacher_outputs[output_name]
        )

    unmasked = make_wrapper(
        mask_by_nulls=False, 
        mask_by_t_threshold=False, 
        p_uncond=0.0 
    )
    assert "classifier_loss" in unmasked.train_step((images, classes))
    unconditional = make_wrapper(
        clf_train_type="uncond", 
        train_cfg_scale=1.0, 
        mask_by_nulls=False 
    )
    unconditional_results = unconditional.train_step((images, classes))
    assert "classifier_loss" in unconditional_results

    ensemble = make_wrapper(
        use_ensemble_loss_instead=True, 
        mask_by_nulls=False 
    )
    assert ensemble.ensemble_loss_fn is not None
    assert ensemble.ensemble_loss_fn.network is ensemble.network
    ensemble_predictions = ensemble.compute_clf_kl_ctr_distil_loss(
        classes, 
        both[3][0], both[5][0], both[4][0], 
        x0=images, 
        training=False 
    )[5]
    assert ensemble_predictions.shape == (2, 2)
    tf.debugging.assert_near(
        tf.reduce_sum(ensemble_predictions, axis=-1), 
        tf.ones(tuple([2])), atol=1e-5
    )
    assert "classifier_loss" in ensemble.test_step((images, classes))

    auxiliary_network = make_network(
        clf_depth=2, 
        clf_vit_block_ids=[], 
        clf_reshaper_ids_dict={1: "flatten", 2: "unflatten"}, 
        clf_reshaper_kwargs={"add_kl": True, "latent_dim_ratio": [1.0]}, 
        clf_cls_token_regularizer_ids=[None], 
        force_global_avg_pooling=True 
    )
    auxiliary = make_wrapper(
        network=auxiliary_network, 
        kl_loss_coef=0.01, 
        ctr_loss_coef=0.01, 
        mask_by_nulls=False 
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
        auxiliary_outputs["clf_z_vals_list"], 
        auxiliary_outputs["clf_regs_list"] 
    )
    assert float(auxiliary_losses[2]) >= 0.0
    assert float(auxiliary_losses[3]) >= 0.0
    auxiliary_metrics = auxiliary.get_clf_results_dict(
        auxiliary_losses[1], classes, 
        auxiliary_losses[5], 
        total_loss=auxiliary_losses[0], 
        clf_kl_loss=auxiliary_losses[2], 
        clf_ctr_loss=auxiliary_losses[3], 
        clf_ctr_preds=auxiliary_losses[6]
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
        vit_block_mlp_ratio=1.0 
    )
    metadata_free = DiffusionClassifier(
        network=raw_network, 
        mask_by_nulls=False, 
        use_ema=False, 
        test_network_name="raw", 
        scheduler_name="linear", 
        test_steps=2 
    )
    assert metadata_free.use_clf_kl_loss is None
    assert metadata_free.use_clf_ctr_loss is None

    progressive = make_wrapper(mask_by_nulls=False)
    progressive_history = progressive.fit_progressively(
        stage_tasks=[
            {"depth": {"classifier": "vision_transformer_block"}}, 
            {"depth": {"classifier": "vision_transformer_block"}}
        ], 
        x=dataset, 
        stages_verbose=False, 
        stage_epochs=1, 
        final_epochs=0, 
        verbose=0 
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
        "classifier": [] 
    })
    assert network_growth["network"]["added"] == 1
    assert network_growth["classifier"]["added"] == 0
    both_growth = branch_growth._add_depths({
        "network": "vision_transformer_block", 
        "classifier": "vision_transformer_block" 
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
        trainable=False, 
        dtype="float64", 
        name="policy_classifier_wrapper" 
    )
    assert policy.name == "policy_classifier_wrapper"
    assert policy.trainable is False
    assert policy.dtype_policy.name == "float64"
    policy_config = policy.get_config()
    assert policy_config["mask_t_percentage"] == 70
    assert policy_config["name"] == "policy_classifier_wrapper"
    assert policy_config["trainable"] is False
    assert tf.keras.dtype_policies.get(policy_config["dtype"]).name == "float64"
    policy_clone = DiffusionClassifier.from_config(policy_config)
    assert policy_clone.network is not policy.network
    assert policy_clone.name == policy.name
    assert policy_clone.dtype_policy.name == "float64"

    for kwargs in (
        {"mask_by_nulls": True, "p_uncond": 0.0}, 
        {"mask_t_percentage": -1, "mask_by_nulls": False}, 
        {"mask_t_percentage": 101, "mask_by_nulls": False}, 
        {"clf_loss_coef": -1., "mask_by_nulls": False}, 
        {"clf_distil_loss_coef": float("nan"), "mask_by_nulls": False}, 
        {"clf_acc_coef": float("inf"), "mask_by_nulls": False}, 
        {"clf_train_type": "unknown", "mask_by_nulls": False}, 
        {
            "clf_train_type": "uncond", "train_cfg_scale": None, 
            "mask_by_nulls": False
        }
    ):
        try:
            DiffusionClassifier(
                network=make_network(), test_steps=2, **kwargs
            )
        except AssertionError:
            pass
        # Invalid classifier masks, coefficients, or branch selectors must be rejected.
        else:
            raise AssertionError(f"Expected invalid classifier wrapper: {kwargs}")
    try:
        wrapper.get_clf_results_dict(
            tf.constant(1.0), classes, both[3][0], 
            use_total_loss=True
        )
    except AssertionError:
        pass
    # Enabled total-loss tracking must reject a missing classifier total.
    else:
        raise AssertionError("Missing requested total loss must fail")
    try:
        wrapper.get_clf_results_dict(
            tf.constant(1.0), classes, both[3][0], 
            use_kl_loss=True
        )
    except AssertionError:
        pass
    # Enabled classifier KL tracking must reject a missing KL scalar.
    else:
        raise AssertionError("Missing requested classifier KL loss must fail")
    try:
        wrapper.get_clf_results_dict(
            tf.constant(1.0), classes, both[3][0], 
            use_ctr_loss=True
        )
    except AssertionError:
        pass
    # Enabled classifier-token tracking must reject missing token-loss inputs.
    else:
        raise AssertionError("Missing requested classifier token loss must fail")

    tf.keras.backend.clear_session()

    return {"DiffusionClassifier": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
