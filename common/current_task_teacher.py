"""Construct independent native teachers for one continual task's new classes."""

from __future__ import annotations

import tensorflow as tf

from collections.abc import Sequence

from copy import deepcopy

from common.runtime import derive_seed

from diffusion import DiffusionModel, DiffusionClassifier, DiffusionClassifierV2
from diffusion.models.wrapper import copy_network_weights_by_layer


def _configure_raw_teacher(
    config: dict[str, object], 
    class_count: int, 
    seed: int | None
) -> dict[str, object]:
    """Copy raw constructor settings with one explicit class width and fresh seeds.

    Composite encoder/decoder constructors retain nested vocabulary and seed
    options, so their saved values must follow the new outer architecture.

    Args:
        config (dict[str, object]): Saved native raw-network constructor settings.
        class_count (int): Positive task-local or full student vocabulary width.
        seed (int | None): Seed for this independent network and its nested branches;
            None leaves every derived branch unseeded.

    Returns:
        dict[str, object]: Detached constructor settings with updated class widths
        and seeds, without changing the source network's configuration.

    Raises:
        ValueError: If seed is outside the supported interval when deriving a nested branch seed.
    """

    result = deepcopy(config)
    result.update(num_classes=class_count, trainable=True, seed=seed)
    for name in ("encoder_kwargs", "decoder_kwargs"):
        nested = result.get(name)
        # Composite branches must share the new outer vocabulary and seed lineage.
        if isinstance(nested, dict):
            result[name] = _configure_raw_teacher(
                nested, 
                class_count, 
                derive_seed(seed, name)
            )

    return result


def student_task_class_ids(
    student: DiffusionModel, 
    class_ids: Sequence[int]
) -> list[int]:
    """Translate dataset task labels into the student's current output columns.

    Args:
        student (DiffusionModel): Native wrapper with its scheduled vocabulary.
        class_ids (Sequence[int]): Dataset labels before CFG condition offsets.

    Returns:
        list[int]: Student output columns in the requested dataset-label order.

    Raises:
        ValueError: A requested label is undiscovered or maps outside the head.
    """

    requested = [int(value) for value in class_ids]
    # Dynamic wrappers may discover dataset labels in a different output order.
    if student.network.dynamic_num_classes:
        # Every requested label must belong to the current student vocabulary.
        if any(value not in student.seen_classes for value in requested):
            raise ValueError("A teacher task cannot cover undiscovered student classes.")
        
        output_ids = [int(student.seen_classes[value]) for value in requested]
    # Fixed heads consume output-column IDs directly.
    else:
        output_ids = requested
    
    # Invalid columns would silently route teacher targets to unrelated classes.
    if any(value < 0 or value >= int(student.network.num_classes) for value in output_ids):
        raise ValueError("A teacher task class is outside the student output vocabulary.")
    
    return output_ids


def make_current_task_teacher(
    student: DiffusionModel, 
    class_ids: Sequence[int], 
    initialization: str = "fresh", 
    seed: int | None = None
) -> tuple[DiffusionModel, list[int]]:
    """Build a task-local fresh teacher or an independent copy of the student.

    Teacher-trained classifier-token losses count as active objectives only when
    the student actually constructs classifier regularizer targets. A coefficient
    and unused regularizer options alone do not justify fitting another model.

    Args:
        student (DiffusionModel): Compiled wrapper after scheduled class expansion.
        class_ids (Sequence[int]): New dataset labels in task schedule order; the
            student's seen_classes mapping determines their output columns.
        initialization (str): ``fresh`` initializes only this task's class vocabulary;
            ``student`` copies the student's full current vocabulary and weights.
            Defaults to ``'fresh'``.
        seed (int | None): Independent network/wrapper seed; None leaves
            initialization and wrapper random streams unseeded.
            Defaults to ``None``.

    Returns:
        tuple[DiffusionModel, list[int]]: Compiled native teacher wrapper and
        output-column-to-student class map. The wrapper follows the student's
        numerical policy and owns independent trainable weights and optimizers.
        Training data retains global IDs; ``seen_classes`` maps them internally.
        The caller owns fitting, freezing, attaching, and releasing this teacher.

    Raises:
        ValueError: Class IDs, initialization, or active current objectives are invalid.
        TypeError: The student is not a supported compiled diffusion wrapper.
    """

    # Teacher construction needs the native architecture and independent compile settings.
    if not isinstance(student, DiffusionModel) or not student.compiled:
        raise TypeError("Current-task teachers require a compiled diffusion wrapper.")
    requested = [int(value) for value in class_ids]
    # Empty or repeated task labels cannot define an unambiguous local head.
    if not requested or len(set(requested)) != len(requested) or min(requested) < 0:
        raise ValueError("Current-task teacher classes must be distinct nonnegative IDs.")
    # Only fresh parameters or an independent student-weight copy are supported.
    if initialization not in ("fresh", "student"):
        raise ValueError("current_teacher_init must be 'fresh' or 'student'.")
    
    source_width = int(student.network.num_classes)
    requested_output_ids = student_task_class_ids(student, requested)

    source_mapping = dict(student.seen_classes) or {
        index: index 
        for index in range(source_width)
    }
    output_class_ids = requested_output_ids if initialization == "fresh" else list(range(source_width))
    teacher_mapping = {value: index for index, value in enumerate(requested)} \
                    if initialization == "fresh" else source_mapping
    raw_config = _configure_raw_teacher(
        student.network.get_config(), 
        len(output_class_ids), 
        seed
    )
    network = type(student.network).from_config(raw_config)

    options = deepcopy(student.get_config())
    for name in (
        "network", "teacher_network", 
        "current_teacher_network", 
        "current_teacher_class_ids", 
        "current_teacher_task_class_ids", 
        "route_controller", "extensions"
    ):
        options.pop(name, None)

    noise_active = float(student.noise_distil_loss_coef) > 0. and float(
        getattr(student, "current_teacher_noise_loss_weight", 1.)
    ) > 0.
    regularizer = getattr(student.network, "clf_cls_token_regularizer_kwargs", None)
    # Classifier-specific token settings take precedence over shared regularizers.
    if regularizer is None:
        regularizer = getattr(student.network, "cls_token_regularizer_kwargs", {})

    classifier_active = isinstance(student, DiffusionClassifier) and (
        float(student.clf_distil_loss_coef) > 0.
        or (float(student.ctr_loss_coef) > 0.
            and bool(getattr(student.network, "clf_cls_token_regularizer_ids", ()))
            and regularizer.get("train_type", "normal") in ("distil", "both"))
    ) and float(getattr(student, "current_teacher_clf_loss_weight", 1.)) > 0.
    # A disabled current objective must not trigger an otherwise unused teacher fit.
    if not (noise_active or classifier_active):
        raise ValueError("Dual-teacher training requires an active current-teacher objective.")
    
    options.update(
        network=network, 
        teacher_network=None, 
        trainable_teacher=False, 
        defer_teacher=False, 
        use_ema=False, 
        test_network_name="raw", 
        swap_noise_image=False, 
        trainable=True, 
        seen_classes=teacher_mapping, 
        noise_loss_coef=(float(student.noise_loss_coef) or 1.) if noise_active else 0., 
        noise_distil_loss_coef=0., 
        image_loss_coef=0., 
        kl_loss_coef=0., 
        ctr_loss_coef=0., 
        seed=seed, 
        dtype=student.dtype_policy
    )
    # Classifier wrappers train supervised probabilities while disabling their own KD.
    if isinstance(student, DiffusionClassifier):
        options.update(
            clf_loss_coef=(float(student.clf_loss_coef) or 1.) if classifier_active else 0., 
            clf_distil_loss_coef=0.
        )
    wrapper_type = DiffusionClassifierV2 if isinstance(student, DiffusionClassifierV2) \
                else DiffusionClassifier if isinstance(student, DiffusionClassifier) else DiffusionModel
    teacher = wrapper_type(**options)
    teacher.set_current_resolution(student.current_resolution[0])

    # Warm initialization copies trained values only after the independent topology is built.
    if initialization == "student":
        copy_network_weights_by_layer(student.network, teacher.network)

    teacher.network.trainable = True
    teacher.compile(**tf.keras.utils.deserialize_keras_object(student.get_compile_config()))
    
    return teacher, output_class_ids


def annotate_teacher_task_classes(
    teacher: tf.keras.Model, 
    student: DiffusionModel, 
    class_ids: Sequence[int]
) -> None:
    """Record which student output columns a completed-task teacher has learned.

    Args:
        teacher (tf.keras.Model): Independent raw snapshot to annotate.
        student (DiffusionModel): Wrapper whose output vocabulary the snapshot uses.
        class_ids (Sequence[int]): Taught classes in the learner's remapped dataset
            vocabulary, without the classifier-free-guidance condition offset.

    Returns:
        None: Installs an untracked tuple of student output IDs on the snapshot.
        Fixed full-width heads therefore exclude still-unseen output columns.

    Raises:
        ValueError: A scheduled class is absent from the student's live vocabulary.
    """

    output_ids = student_task_class_ids(student, class_ids)
    object.__setattr__(teacher, "_diffusion_task_class_ids", tuple(output_ids))
