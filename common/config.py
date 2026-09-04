"""Dataclass configuration tree and YAML serialization for experiments.

Legacy DiT sections expose :meth:`KwargsMixin.kwargs`. Generic model families
use the compact ``name``/``kwargs`` fields consumed by :mod:`common.train` and
the shared HPO runner. ``ContinuallyLearnConfig`` contains the policy unique to
class-incremental runs while the other sections remain reusable.
"""

from __future__ import annotations

import os

import random

import yaml

from dataclasses import (
    MISSING, 
    Field, 
    asdict, 
    dataclass, 
    field, 
    fields, 
    is_dataclass
)
from typing import Any, TextIO, TypeVar
from collections.abc import Mapping


SectionT = TypeVar("SectionT")

_TRAINING_TASKS = frozenset({
    "legacy", "generation", "joint", 
    "classification", "continual"
})
"""Task selectors implemented by the shared experiment pipeline."""


def _reject_duplicate_yaml_keys(
    loader: yaml.SafeLoader, 
    node: yaml.Node, 
    visited: set[int] | None = None
) -> None:
    """Reject duplicate explicit keys anywhere in one parsed YAML document.

    YAML merge keys retain their standard precedence semantics: collisions
    introduced by ``<<`` are permitted, while duplicate keys written directly
    in the same mapping are rejected.  Key comparison uses the same safely
    constructed Python values that PyYAML would use for the resulting mapping.

    Args:
        loader (yaml.SafeLoader): Safe loader that parsed ``node``.
        node (yaml.Node): Scalar, sequence, or mapping node to validate.
        visited (set[int] | None): Identity set used to terminate recursive
            YAML alias graphs. Callers normally omit it.

    Returns:
        None: Every reachable mapping has unique explicit keys.

    Raises:
        yaml.constructor.ConstructorError: If a mapping repeats an explicit
            key or contains an unhashable key.
    """

    # Anchors may create recursive node graphs; inspect each node only once.
    if visited is None:
        visited = set()
    node_id = id(node)
    # A previously inspected alias target cannot introduce a new local key.
    if node_id in visited:
        return
    visited.add(node_id)

    # Mapping nodes need local duplicate detection plus recursive validation.
    if isinstance(node, yaml.MappingNode):
        seen: dict[object, yaml.Node] = {}
        for key_node, value_node in node.value:
            # Merge keys intentionally inherit YAML's ordinary override rules.
            if key_node.tag != "tag:yaml.org,2002:merge":
                key = loader.construct_object(key_node, deep=True)
                try:
                    duplicate = key in seen
                except TypeError as error:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        "found an unhashable key",
                        key_node.start_mark,
                    ) from error

                # Silent last-value-wins behavior can invalidate experiments.
                if duplicate:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r}",
                        key_node.start_mark,
                    )
                seen[key] = key_node

            _reject_duplicate_yaml_keys(loader, key_node, visited)
            _reject_duplicate_yaml_keys(loader, value_node, visited)
        return

    # Sequence children can themselves contain mappings with duplicate keys.
    if isinstance(node, yaml.SequenceNode):
        for child_node in node.value:
            _reject_duplicate_yaml_keys(loader, child_node, visited)


def _safe_load_unique_yaml(stream: TextIO) -> object:
    """Safely construct one YAML document after duplicate-key validation.

    Args:
        stream (TextIO): Open UTF-8 text stream containing one YAML document.

    Returns:
        object: Safely constructed YAML value, or ``None`` for an empty file.

    Raises:
        yaml.YAMLError: If syntax, document count, keys, or construction are
            invalid.
    """

    loader = yaml.SafeLoader(stream)
    try:
        node = loader.get_single_node()
        # Preserve ``yaml.safe_load`` semantics for an empty document.
        if node is None:
            return None
        _reject_duplicate_yaml_keys(loader, node)
        return loader.construct_document(node)
    finally:
        loader.dispose()


def normalize_training_task(task: object) -> str:
    """Return one canonical task selector used across common-layer APIs.

    Args:
        task (object): Candidate task name. Supported strings are ``"legacy"``,
            ``"generation"``, ``"joint"``, ``"classification"``, and
            ``"continual"``; spelling is case-insensitive.

    Returns:
        str: Lowercase supported task selector.

    Raises:
        ValueError: If the normalized task is unsupported.
    """

    normalized = task.lower()
    # Reject unknown tasks before dataset/model construction can misroute them.
    if normalized not in _TRAINING_TASKS:
        raise ValueError(
            "training task must be one of " + str(sorted(_TRAINING_TASKS)) + "."
        )

    return normalized


def _all_depth_ids() -> list[int | None]:
    """Return the default marker selecting every eligible network depth.

    Returns:
        list[int | None]: A new ``[None]`` list for one dataclass instance.
    """

    return [None]


def _default_regularizer_range() -> dict[str, object]:
    """Return the default regularizer interval and training modes.

    Returns:
        dict[str, object]: A new mapping selecting the first token, normal
        labels, and hard teacher targets when distillation is requested.
    """

    return {
        "start": 0, 
        "end": 1, 
        "train_type": "normal", 
        "distil_type": "hard"
    }


def _default_feature_aggregation() -> dict[int, list[int]]:
    """Return the default final-depth feature route.

    Returns:
        dict[int, list[int]]: A new mapping from classifier depth 1 to the
        source model's final depth.
    """

    return {1: [-1]}


def _default_classifier_connection() -> dict[int, list[int]]:
    """Return the default route from the final source classifier depth.

    Returns:
        dict[int, list[int]]: A new ``{-1: [-1]}`` mapping.
    """

    return {-1: [-1]}


def _default_vae_hiddens() -> list[int]:
    """Return the default single dense VAE hidden width.

    Returns:
        list[int]: A new ``[16]`` list.
    """

    return [16]


def _default_unet_widths() -> list[int]:
    """Return the standard three-stage U-Net widths.

    Returns:
        list[int]: A new ``[32, 64, 96]`` list.
    """

    return [32, 64, 96]


def _default_buffer_kwargs() -> dict[str, object]:
    """Return default fixed replay-buffer controls.

    Returns:
        dict[str, object]: New capacity, sampling, insertion, seed, and FIFO
        strategy values.
    """

    return {
        "maxlen": 10_000, 
        "sample_num": 1_000, 
        "insert_num": 1_000, 
        "seed": None, 
        "strategy": "fifo"
    }


def _default_generative_replay_kwargs() -> dict[str, int]:
    """Return default generative-replay sample counts.

    Returns:
        dict[str, int]: New training and per-class generation counts.
    """

    return {"train_num": 1_000, "samples_per_class": 1_000}


def resolve_continual_schedule(
    class_num: int | None, 
    class_order: list[int] | tuple[int, ...] | None = None, 
    task_groups: list[list[int]] | tuple[tuple[int, ...], ...] | None = None, 
    available_class_num: int | None = None, 
    task_size: int = 1, 
    class_order_mode: str = "fixed", 
    task_order_mode: str = "fixed", 
    seed: int | None = None
) -> tuple[list[int], list[list[int]]]:
    """Validate and resolve a class-incremental experiment schedule.

    ``class_order`` and ``task_groups`` use the dataset's original class IDs.
    When both are supplied, flattening ``task_groups`` must reproduce
    ``class_order`` exactly. Without explicit groups, classes are partitioned
    into consecutive groups of ``task_size``; the default therefore starts
    with one class and adds one class per task. Class order and whole-task order
    can be shuffled independently by the same configured seed.

    Args:
        class_num (int | None): Number of selected classes. ``None`` infers the
            count from an explicit schedule or ``available_class_num``.
        class_order (Sequence[int] | None): Ordered original dataset labels.
        task_groups (Sequence[Sequence[int]] | None): Nonempty groups introduced
            at successive tasks.
        available_class_num (int | None): Optional dataset class count used to
            validate label bounds and fill a completely unspecified schedule.
        task_size (int): Positive number of classes placed in each automatically
            constructed task. The final task may contain fewer classes.
        class_order_mode (str): ``"fixed"`` preserves the supplied/natural class
            order; ``"random"`` shuffles it reproducibly before grouping.
        task_order_mode (str): ``"fixed"`` preserves task order; ``"random"``
            shuffles complete groups without changing within-task class order.
            An automatic short remainder is never placed first.
        seed (int | None): Seed used for class/task shuffling.

    Returns:
        tuple[list[int], list[list[int]]]: Resolved class order and task groups.

    Raises:
        ValueError: If the schedule is empty, duplicated, inconsistent, out of
            range, or uses an unsupported ordering mode.
    """

    class_order_mode = str(class_order_mode).lower()
    task_order_mode = str(task_order_mode).lower()
    seed = None if seed is None else int(seed)
    task_size = int(task_size)
    if task_size < 1:
        raise ValueError("task_size must be positive.")
    # Restrict class ordering to the two documented protocols.
    if class_order_mode not in ("fixed", "random"):
        raise ValueError("class_order_mode must be 'fixed' or 'random'.")
    # Restrict task ordering independently from within-task ordering.
    if task_order_mode not in ("fixed", "random"):
        raise ValueError("task_order_mode must be 'fixed' or 'random'.")

    # Normalize explicit groups before using them to infer the class order.
    normalized_groups = None
    # Materialize caller-provided task groups for validation and safe reuse.
    if task_groups is not None:
        normalized_groups = [list(group) for group in task_groups]
        # Reject empty schedules and empty individual experiences.
        if not normalized_groups or any(not group for group in normalized_groups):
            raise ValueError("task_groups must contain only nonempty groups.")
        # Explicit membership and random class ordering are contradictory.
        if class_order_mode == "random":
            raise ValueError(
                "class_order_mode='random' cannot be combined with task_groups."
            )
    automatic_task_groups = normalized_groups is None

    # Infer the order from explicit groups when no separate order was supplied.
    grouped_order = [label for group in normalized_groups for label in group] \
        if normalized_groups is not None else None
    # Let task groups define the class order when it was not separately given.
    if class_order is None:
        resolved_order = grouped_order
    # Otherwise preserve the explicit order and verify redundant group metadata.
    else:
        resolved_order = list(class_order)
        # Keep redundant schedule descriptions consistent and reproducible.
        if grouped_order is not None and resolved_order != grouped_order:
            raise ValueError(
                "Flattened task_groups must exactly match class_order."
            )

    # Fill a completely unspecified schedule from the requested dataset prefix.
    if resolved_order is None:
        resolved_count = available_class_num if class_num is None else class_num
        # Require some source of truth for the number of classes to schedule.
        if resolved_count is None:
            raise ValueError(
                "class_num, class_order, task_groups, or available_class_num "
                "must define the continual schedule."
            )
        resolved_order = list(range(resolved_count))

    resolved_order = [int(label) for label in resolved_order]
    if normalized_groups is not None:
        normalized_groups = [
            [int(label) for label in group]
            for group in normalized_groups
        ]
    # Disallow negative labels, which no supported dataset uses.
    if any(label < 0 for label in resolved_order):
        raise ValueError("Continual class IDs must be nonnegative.")
    # Enforce the known upper label bound for built-in datasets.
    if available_class_num is not None and any(
        label >= available_class_num for label in resolved_order
    ):
        raise ValueError(
            "Continual class IDs must be smaller than the dataset class count."
        )
    # Prevent the same class from being introduced by multiple tasks.
    if len(set(resolved_order)) != len(resolved_order):
        raise ValueError("Continual class IDs must be unique.")

    selected_class_num = len(resolved_order)
    # Continual evaluation requires at least one transition between tasks.
    if selected_class_num < 2:
        raise ValueError("A continual schedule must contain at least two classes.")
    # Keep an explicit count consistent with the resolved label schedule.
    if class_num is not None and class_num != selected_class_num:
        raise ValueError("class_num must equal the number of scheduled classes.")
    # Keep selected-class cardinality within the built-in dataset capacity.
    if available_class_num is not None and selected_class_num > available_class_num:
        raise ValueError("The continual schedule exceeds the dataset class count.")

    schedule_rng = random.Random(seed)
    # Randomize class identity before automatic grouping when requested.
    if normalized_groups is None:
        # Shuffle individual labels only for the class-random protocol.
        if class_order_mode == "random":
            schedule_rng.shuffle(resolved_order)
        normalized_groups = [
            resolved_order[index:index + task_size]
            for index in range(0, len(resolved_order), task_size)
        ]

    if len(normalized_groups) < 2:
        raise ValueError(
            "A continual schedule must contain at least two task groups."
        )

    # Reorder complete tasks while preserving class order inside each group.
    if task_order_mode == "random":
        remainder_group = None
        # An automatic short final group may move, but task one must still
        # contain the requested task_size classes.
        if automatic_task_groups and len(normalized_groups[-1]) < task_size:
            remainder_group = normalized_groups.pop()
        schedule_rng.shuffle(normalized_groups)
        if remainder_group is not None:
            insertion_index = schedule_rng.randrange(
                1,
                len(normalized_groups) + 1,
            )
            normalized_groups.insert(insertion_index, remainder_group)

    # The returned order always describes the actual resolved task stream.
    resolved_order = [label for group in normalized_groups for label in group]

    return resolved_order, normalized_groups


class KwargsMixin:
    """Convert a configuration dataclass into constructor keyword arguments."""

    def kwargs(self: KwargsMixin) -> dict[str, Any]:
        """Return a recursively copied dictionary of all dataclass fields.

        Returns:
            dict[str, Any]: Every field, including values equal to defaults and
            ``None``.  Keys are not renamed or filtered before callers expand
            the result with ``**``.
        """

        return asdict(self)


@dataclass
class DiffusionTransformerConfig(KwargsMixin):
    """Arguments forwarded to ``DiffusionTransformer``."""

    num_classes: int | None = 10
    use_cfg: bool = True
    timesteps: int = 1_000
    image_size: int = 28
    channels: int = 1
    patch_size: int = 2
    dim: int = 32
    dim_forced: bool = True
    patchify_with_cnn: bool = False
    patches_pos_embed_type: str = "2d_sincos"
    patches_pos_merger_type: str = "add"
    patches_conds_merger_type: str | None = None
    shift_inputs: bool = False
    cond_dim: int | None = None
    cond_type: str | None = "time_label"
    conds_merger_type: str = "add"
    time_embed_type: str = "1d_sincos"
    time_freq_dim: int | None = None
    time_embed_trainable: bool = False
    time_mlp_ratio: float | None = None
    label_embed_type: str = "new_weight"
    label_embed_trainable: bool = False
    label_freq_dim: int | None = None
    label_mlp_ratio: float | None = None
    cls_token_type: str | None = None
    cls_token_freq_dim: int | None = None
    cls_token_mlp_ratio: float | None = None
    cls_token_pos_merger_type: str = "add"
    distil_token_type: str | None = None
    distil_token_freq_dim: int | None = None
    distil_token_mlp_ratio: float | None = None
    distil_token_pos_merger_type: str = "add"
    depth: int = 2
    connection_ids_dict: dict = field(default_factory=dict)
    connection_kwargs: dict = field(default_factory=dict)
    cross_attention_ids_dict: dict = field(default_factory=dict)
    cross_attention_kwargs: dict = field(default_factory=dict)
    cross_attention_plug_type: str = "values"
    vit_block_ids: list[int | None] = field(default_factory=_all_depth_ids)
    use_decoder_ids: list[int | None] = field(default_factory=list)
    mha_key_dim: int | None = None
    mha_value_dim: int | None = None
    mha_num_heads: int = 4
    vit_block_mlp_ratio: float = 4.0
    vit_block_mlp_output_dims: dict[int, int] = field(default_factory=dict)
    ln_mlp_ratio: float | None = None
    ln_no_adaptation: bool = False
    drop_prob: float = 0.0
    drop_per_sample: bool = True
    local_mixer_ids: list[int | None] = field(default_factory=list)
    local_mixer_kwargs: dict = field(default_factory=dict)
    downsample_ids: list[int | None] = field(default_factory=list)
    downsample_kwargs: dict = field(default_factory=dict)
    upsample_ids: list[int | None] = field(default_factory=list)
    upsample_kwargs: dict = field(default_factory=dict)
    reshaper_ids_dict: dict[int, str] = field(default_factory=dict)
    reshaper_kwargs: dict = field(default_factory=dict)
    cls_token_regularizer_ids: list[int | None] = field(default_factory=list)
    cls_token_regularizer_kwargs: dict = field(
        default_factory=_default_regularizer_range
    )
    final_ffn_activation_func: str = "linear"
    use_refiner_cnn: bool = False
    refiner_cnn_hidden_dim: int | None = None
    refiner_cnn_residual: bool = True
    final_activation_func: str = "linear"
    use_unpatchify: bool = True
    name_prefix: str = ""
    build: bool = True


@dataclass
class DiTClassifierConfig(DiffusionTransformerConfig):
    """Arguments forwarded to ``DiTClassifier``, including inherited ones."""

    aggregate_from_noises: bool = False
    feature_aggregation_ids_dict: dict = field(
        default_factory=_default_feature_aggregation
    )
    feature_aggregation_kwargs: dict = field(default_factory=dict)
    cross_attention_aggregation_ids_dict: dict = field(default_factory=dict)
    cross_attention_aggregation_kwargs: dict = field(default_factory=dict)
    classifier_only_cls_token: bool = True
    classifier_only_distil_token: bool = True
    clf_dim: int | None = None
    clf_dim_forced: bool = False
    clf_cond_type: str | None = "time_label"
    clf_cls_token_type: str | None = "new_weight"
    clf_distil_token_type: str | None = None
    clf_depth: int = 1
    clf_connection_ids_dict: dict = field(
        default_factory=_default_classifier_connection
    )
    clf_connection_kwargs: dict | None = None
    clf_cross_attention_ids_dict: dict = field(default_factory=dict)
    clf_cross_attention_kwargs: dict | None = None
    clf_cross_attention_plug_type: str | None = None
    clf_vit_block_ids: list[int | None] = field(default_factory=_all_depth_ids)
    clf_use_decoder_ids: list[int | None] = field(default_factory=list)
    clf_mha_key_dim: int | None = None
    clf_mha_value_dim: int | None = None
    clf_mha_num_heads: int | None = None
    clf_vit_block_mlp_ratio: float | None = None
    clf_vit_block_mlp_output_dims: dict[int, int] | None = None
    clf_ln_mlp_ratio: float | None = None
    clf_ln_no_adaptation: bool | None = None
    clf_drop_prob: float | None = None
    clf_drop_per_sample: bool | None = None
    clf_local_mixer_ids: list[int | None] = field(default_factory=list)
    clf_local_mixer_kwargs: dict | None = None
    clf_downsample_ids: list[int | None] = field(default_factory=list)
    clf_downsample_kwargs: dict | None = None
    clf_upsample_ids: list[int | None] = field(default_factory=list)
    clf_upsample_kwargs: dict | None = None
    clf_reshaper_ids_dict: dict[int, str] = field(default_factory=dict)
    clf_reshaper_kwargs: dict | None = None
    clf_cls_token_regularizer_ids: list[int | None] = field(default_factory=list)
    clf_cls_token_regularizer_kwargs: dict | None = None
    force_global_avg_pooling: bool = False
    classifier_mlp_ratio: int | None = None
    classifier_mlp_activation_func: str = "tanh"
    dropout_rate: float = 0.0


@dataclass
class DiTDecoderConfig(DiffusionTransformerConfig):
    """Arguments forwarded to ``DiTDecoder``."""

    encoder_output_grid_size: int | None = None
    encoder_output_dim: int | None = None
    encoder_feature_grid_sizes: list[int | None] | None = None
    encoder_feature_dims: list[int] | None = None
    shift_inputs: bool = True
    use_decoder_ids: list[int | None] = field(default_factory=_all_depth_ids)
    decoder_separate_cond: bool = False
    use_causal_mask: bool = True
    feature_aggregation_ids_dict: dict = field(default_factory=dict)
    feature_aggregation_kwargs: dict = field(default_factory=dict)
    cross_attention_aggregation_ids_dict: dict = field(default_factory=dict)
    cross_attention_aggregation_kwargs: dict = field(default_factory=dict)


@dataclass
class DiTEncoderDecoderConfig(DiffusionTransformerConfig):
    """Arguments forwarded to ``DiTEncoderDecoder``."""

    encoder_kwargs: dict[str, object] | None = None
    decoder_kwargs: dict[str, object] | None = None


@dataclass
class DiTEncoderDecoderClassifierConfig(DiTClassifierConfig):
    """Arguments forwarded to ``DiTEncoderDecoderClassifier``."""

    encoder_kwargs: dict[str, object] | None = None
    decoder_kwargs: dict[str, object] | None = None


@dataclass
class UNetConfig(KwargsMixin):
    """Arguments forwarded to ``UNet``."""

    num_classes: int | None = 10
    use_cfg: bool = True
    timesteps: int = 1_000
    image_size: int = 32
    channels: int = 1
    widths: list[int] = field(default_factory=_default_unet_widths)
    block_depth: int = 2
    bottleneck_width: int = 128
    bottleneck_depth: int = 2
    image_embedding_dim: int = 21
    time_embedding_dim: int = 22
    label_embedding_dim: int = 21
    activation_func: str = "swish"
    final_activation_func: str = "linear"
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
    downsampling_method: str = "avg_pooling"
    upsampling_method: str = "interpolate"
    upsampling_interpolation: str = "bilinear"
    use_skip_connections: bool | None = None
    reshaper_ids_dict: dict[int, str] = field(default_factory=dict)
    reshaper_kwargs: dict = field(default_factory=dict)
    cls_token_regularizer_ids: list[int | None] = field(default_factory=list)
    cls_token_regularizer_kwargs: dict = field(
        default_factory=_default_regularizer_range
    )
    extra_depth_specs: list[object] = field(default_factory=list)
    name_prefix: str = ""
    build: bool = True


@dataclass
class UNetClassifierConfig(UNetConfig):
    """Arguments forwarded to ``UNetClassifier``."""

    aggregate_from_noises: bool = False
    feature_aggregation_ids_dict: dict = field(
        default_factory=_default_feature_aggregation
    )
    classifier_only_cls_token: bool = False
    classifier_only_distil_token: bool = False
    clf_dim: int | None = None
    clf_depth: int = 1
    clf_block_depth: int = 1
    clf_reshaper_kwargs: dict = field(default_factory=dict)
    clf_cls_token_regularizer_ids: list[int | None] = field(default_factory=list)
    force_global_avg_pooling: bool = True
    classifier_mlp_ratio: float | None = None
    classifier_mlp_activation_func: str = "tanh"


@dataclass
class DiffusionModelConfig(KwargsMixin):
    """Arguments forwarded to ``DiffusionModel`` except ``network``.

    ``seen_classes`` is normally empty in input configs. Dynamic diffusion
    checkpoint saving fills it with the live dataset-label-to-zero-based-target
    mapping. On a continual reload, the raw constructor still receives
    ``num_classes=None``
    and the wrapper uses this mapping to restore the grown topology.
    ``test_steps=None`` lets the model factory cap the default at the raw
    network's timestep count. For backward compatibility,
    ``test_network_name="ema"`` resolves to the raw network when
    ``use_ema=False``.
    """

    use_ema: bool = True
    test_network_name: str = "ema"
    ema_decay: float = 0.999
    scheduler_name: str = "clipped_cosine"
    modify_first_t: bool = False
    p_uncond: float = 0.1
    train_cfg_scale: float | None = None
    test_cfg_scale: float = 4.0
    test_steps: int | None = None
    test_eta: float = 0.0
    noise_loss_coef: float = 1.0
    noise_distil_loss_coef: float = 0.0
    show_separate_noise_losses: bool = False
    image_loss_coef: float = 0.0
    kl_loss_coef: float = 0.0
    ctr_loss_coef: float = 0.0
    kl_train_type: str = "cond"
    ctr_train_type: str = "cond"
    train_noisified_min_timesteps: int = 0
    train_noisified_max_timesteps: int | None = -1
    test_noisified_min_timesteps: int = 0
    test_noisified_max_timesteps: int | None = -1
    resize_method: str = "area"
    resize_antialias: bool = True
    swap_noise_image: bool = False
    map_preprocess: bool = False
    map_num_parallel_calls: int | None = 1
    seen_classes: dict[object, int] = field(default_factory=dict)
    seed: int | None = None
    defer_teacher: bool = False


@dataclass
class DiffusionClassifierConfig(DiffusionModelConfig):
    """Arguments forwarded to ``DiffusionClassifier`` except ``network``.

    ``mask_by_nulls=None`` lets the model factory match the raw network's CFG
    setting. An explicit boolean is forwarded unchanged. Knowledge
    distillation keeps the historical hard/T=1/all-sample defaults;
    ``clf_distil_temperature`` controls soft-target smoothing and
    ``clf_distil_scope``
    selects old classes, replay rows, or all current/replay rows.
    """

    clf_distil_type: str = "hard"
    clf_distil_temperature: float = 1.0
    clf_distil_scope: str = "current_and_replay"
    mask_by_nulls: bool | None = None
    mask_by_t_threshold: bool = False
    mask_t_percentage: int = 70
    use_ensemble_loss_instead: bool = False
    clf_train_type: str = "cond"
    clf_loss_coef: float = 8.6e-3
    clf_distil_loss_coef: float = 0.0
    clf_acc_coef: float = 0.5
    clf_distil_acc_coef: float = 0.5
    ctr_acc_coef: float = 0.0


@dataclass
class DiffusionClassifierV2Config(DiffusionClassifierConfig):
    """Arguments forwarded to ``DiffusionClassifierV2`` except ``network``."""

    mask_by_nulls: bool = False
    clf_loss_coef: float = 1.0
    clf_vars_embedding_ids: list[int] = field(default_factory=list)
    clf_vars_noise_part_ids: list[int] = field(default_factory=list)
    clf_train_noisified_max_timesteps: int | None = None
    clf_test_noisified_max_timesteps: int | None = None


@dataclass
class VariationalAutoencoderConfig(KwargsMixin):
    """Arguments forwarded to ``VariationalAutoencoder``."""

    data_dim: int = 2_048
    latent_dim: int = 8
    hiddens_dims: list[int] = field(default_factory=_default_vae_hiddens)
    hiddens_kwargs: dict = field(default_factory=dict)
    last_activation: str | None = "tanh"
    beta: float = 0.25
    conditioned: bool = False
    class_num: int | None = None
    compile: bool = True
    compile_args: dict = field(default_factory=dict)


@dataclass
class VAEClassifierConfig(KwargsMixin):
    """Arguments forwarded to ``VAEClassifier`` except classifier inputs."""

    data_dim: int = 2_048
    latent_dim: int = 8
    hiddens_dims: list[int] = field(default_factory=_default_vae_hiddens)
    hiddens_kwargs: dict = field(default_factory=dict)
    last_activation: str | None = "tanh"
    beta: float = 0.25
    alpha: float = 1.0
    compile_args: dict = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Dataset selection, preprocessing, batching, and optional trial limits.

    Attributes:
        name (str): ``"mnist"``, ``"fmnist"``, ``"cifar10"``, or
            ``"cifar100"``.
        preprocess (str | None): ``"min-max"``, ``"normalize"``,
            ``"standardize"``/``"diffusion"``, or no scaling. ``None`` is
            resolved automatically for diffusion and VAE model families.
        indices (list[int] | None): Class IDs retained by ordinary dataset
            construction. Continual selection instead follows
            ``continually_learn.class_order`` and ``task_groups``.
        validation_ratio (float): Fraction of training rows reserved for a
            stratified validation split; ``0`` disables the split.
        features_path (str | None): Base path of an optional saved feature
            archive.
        return_features (bool): Use saved features instead of raw images.
        onehot_labels (bool): Return full-width categorical labels instead of
            sparse IDs.
        max_train_samples (int | None): Optional positive training-row limit.
            Continual runs apply a class-preserving limit once before per-task
            selection, so the value must cover every represented class.
        max_val_samples (int | None): Optional positive evaluation-row limit.
            Limits preserve represented classes and apply only to an explicit
            validation split, never to the test set.
        batch_size (int): Positive examples per batch; defaults to 128.
        shuffle_buffer (int): Training shuffle-buffer capacity, including each
            continual task. Values above zero enable shuffling; ``0`` disables
            it. Defaults to 10,000.
        pad (int): Symmetric zero-padding applied to raw images. Continual runs
            apply it before replay; saved features and pretrained/hp-tuned
            classifiers do not support it.
        trainset_len (int | None): Number of prepared training batches,
            populated by ``get_datasets`` for optimizer scheduling.
    """

    name: str = "mnist"
    preprocess: str | None = None
    indices: list[int] | None = None
    validation_ratio: float = 0.
    features_path: str | None = None
    return_features: bool = False
    onehot_labels: bool = False
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    batch_size: int = 128
    shuffle_buffer: int = 10_000
    pad: int = 0
    trainset_len: int | None = None


@dataclass
class ModelConfig:
    """Select the raw network, wrapper, visibility, and initial weights.

    Attributes:
        name (str | None): Generic raw-model/family name. ``None`` retains the
            legacy compact DiT selection below.
        kwargs (dict): Generic raw-model or classifier constructor arguments.
            When nonempty, these retain precedence over the typed section for
            ``name``. Continual diffusion construction overrides
            ``num_classes`` with ``None``.
        wrapper_kwargs (dict): Generic diffusion-wrapper arguments.
        classifier_name (str | None): Target classifier for continual runs.
        classifier_kwargs (dict): Target-classifier architecture arguments.
        loss_function (str): Generative reconstruction/noise loss passed to
            Keras compilation; ``"mse"`` by default.
        with_classifier (bool): Build ``DiTClassifier`` inside
            ``DiffusionClassifier`` when true; otherwise build
            ``DiffusionTransformer`` inside ``DiffusionModel``.
        show_network_summary (bool): Print the wrapper/network summary after
            construction.
        weights_path (str | None): Keras weights file or TensorFlow checkpoint
            prefix loaded after construction, or ``None`` for fresh weights.
            In continual runs it initializes a
            VAE replay model, or the classifier and its incremental head
            prefixes for classifier-only and buffer-based runs. A continual
            diffusion checkpoint requires a paired config containing the
            current ``num_classes`` and zero-based wrapper ``seen_classes``
            mapping. The
            continual factory still constructs a dynamic raw network and the
            wrapper grows it before loading these weights. Training updates
            this field to the saved weight artifact path.
        diffusion_transformer (DiffusionTransformerConfig): Raw denoising
            network settings used only when ``with_classifier=False``.
        dit_classifier (DiTClassifierConfig): Raw joint network settings used
            only when ``with_classifier=True``.
        diffusion_model (DiffusionModelConfig): Process-wrapper settings used
            only when ``with_classifier=False``.
        diffusion_classifier (DiffusionClassifierConfig): Classifier wrapper
            settings used only when ``with_classifier=True``.
        Other typed fields provide constructor settings for their matching
            ``name`` or ``wrapper_name`` selection.
    """

    with_classifier: bool = True
    show_network_summary: bool = True
    weights_path: str | None = None
    loss_function: str = "mse"

    # Generic selections used by config-driven experiments and HPO. ``None``
    # keeps the original DiT/DiT-classifier path above fully backward compatible.
    name: str | None = None
    wrapper_name: str | None = None
    kwargs: dict = field(default_factory=dict)
    wrapper_kwargs: dict = field(default_factory=dict)
    classifier_name: str | None = None
    classifier_kwargs: dict = field(default_factory=dict)

    diffusion_transformer: DiffusionTransformerConfig = field(
        default_factory=DiffusionTransformerConfig
    )
    dit_classifier: DiTClassifierConfig = field(
        default_factory=DiTClassifierConfig
    )
    dit_decoder: DiTDecoderConfig = field(default_factory=DiTDecoderConfig)
    dit_encoder_decoder: DiTEncoderDecoderConfig = field(
        default_factory=DiTEncoderDecoderConfig
    )
    dit_encoder_decoder_classifier: DiTEncoderDecoderClassifierConfig = field(
        default_factory=DiTEncoderDecoderClassifierConfig
    )
    unet: UNetConfig = field(default_factory=UNetConfig)
    unet_classifier: UNetClassifierConfig = field(
        default_factory=UNetClassifierConfig
    )
    variational_autoencoder: VariationalAutoencoderConfig = field(
        default_factory=VariationalAutoencoderConfig
    )
    vae_classifier: VAEClassifierConfig = field(
        default_factory=VAEClassifierConfig
    )
    diffusion_model: DiffusionModelConfig = field(
        default_factory=DiffusionModelConfig
    )
    diffusion_classifier: DiffusionClassifierConfig = field(
        default_factory=DiffusionClassifierConfig
    )
    diffusion_classifier_v2: DiffusionClassifierV2Config = field(
        default_factory=DiffusionClassifierV2Config
    )

    def __post_init__(self: ModelConfig) -> None:
        """Convert nested model-section mappings to typed dataclass instances.

        Returns:
            None: The nested attributes are normalized in place.  Existing
            instances are preserved; mappings are expanded as constructor
            keywords and may omit fields to use their defaults.
        """

        self.diffusion_transformer = _section(self.diffusion_transformer, DiffusionTransformerConfig)
        self.dit_classifier = _section(self.dit_classifier, DiTClassifierConfig)
        self.dit_decoder = _section(self.dit_decoder, DiTDecoderConfig)
        self.dit_encoder_decoder = _section(
            self.dit_encoder_decoder, DiTEncoderDecoderConfig
        )
        self.dit_encoder_decoder_classifier = _section(
            self.dit_encoder_decoder_classifier,
            DiTEncoderDecoderClassifierConfig,
        )
        self.unet = _section(self.unet, UNetConfig)
        self.unet_classifier = _section(
            self.unet_classifier, UNetClassifierConfig
        )
        self.variational_autoencoder = _section(
            self.variational_autoencoder, VariationalAutoencoderConfig
        )
        self.vae_classifier = _section(
            self.vae_classifier, VAEClassifierConfig
        )
        self.diffusion_model = _section(self.diffusion_model, DiffusionModelConfig)
        self.diffusion_classifier = _section(self.diffusion_classifier, DiffusionClassifierConfig)
        self.diffusion_classifier_v2 = _section(
            self.diffusion_classifier_v2, DiffusionClassifierV2Config
        )


@dataclass
class OptimizerConfig:
    """Optimizer and learning-rate schedule settings.

    Attributes:
        initial_learning_rate (float): Initial cosine-decay learning rate;
            defaults to ``5e-3``.
        decay_steps (int | None): Positive cosine-decay duration in optimizer
            steps.  ``None`` is replaced during model construction with
            ``epochs * trainset_len``.
        name (str): ``"adam"``, ``"adamw"``, ``"nadam"``, ``"rmsprop"``,
            or ``"sgd"``.
        schedule (str | None): ``"cosine"``, ``"constant"``, or ``None``.
        weight_decay (float | None): Optional AdamW-style weight decay; nonzero
            values require ``name="adamw"`` under TensorFlow 2.10.
        momentum (float): Momentum used by RMSprop/SGD.
        clipnorm (float | None): Optional positive finite norm used to clip each
            variable's gradient tensor independently. This is Keras
            ``clipnorm`` semantics, not global-gradient clipping.
    """

    initial_learning_rate: float = 5e-3
    decay_steps: int | None = None
    name: str = "adam"
    schedule: str = "cosine"
    weight_decay: float | None = None
    momentum: float = 0.0
    clipnorm: float | None = None


@dataclass
class ContinuallyLearnConfig(KwargsMixin):
    """Class-incremental learning and replay settings.

    Dataset selection and preprocessing remain in :class:`DatasetConfig`,
    model construction remains in :class:`ModelConfig`, and fit/reporting
    controls remain in :class:`TrainingConfig` and :class:`ReportingConfig`.
    This section contains only settings specific to the continual loop.

    Attributes:
        class_num (int | None): Number of classes introduced across the run.
            ``None`` uses the selected dataset's complete class count; an
            explicit value must be between 2 and that count. This controls the
            task sequence, not the dynamic diffusion model's initial head width.
        class_order (list[int] | None): Optional ordering of original dataset
            labels. ``None`` preserves the natural zero-based order.
        task_groups (list[list[int]] | None): Optional nonempty groups of
            original labels introduced per task. Flattening the groups must
            equal ``class_order`` when both are supplied. ``None`` preserves
            automatic grouping by ``task_size``.
        task_size (int): Positive number of classes per automatically built
            task. The default ``1`` starts with one class and adds one class at
            every task.
        class_order_mode (str): ``"fixed"`` or seeded ``"random"`` ordering
            before automatic grouping.
        task_order_mode (str): ``"fixed"`` or seeded ``"random"`` ordering of
            complete groups while preserving order inside each group.
        seed (int | None): Master continual seed used by schedule resolution,
            data, model initialization, replay, training and sampling. ``None``
            falls back to ``training.seed`` for backward compatibility.
        remove_prev_classes (bool): Train later tasks on only the newly
            introduced class when true; otherwise train on every seen class.
        keep_same_model (bool): Carry shared classifier weights and old output
            columns into the next expanded classifier.
        use_loaded_opt (bool): Reconstruct a fresh optimizer from the
            configuration stored with the classifier template. Optimizer slots
            and its iteration counter are intentionally not reused by
            :func:`common.model.get_model`.
        use_buffer (bool): Use fixed-size sample replay instead of a generative
            replay model.
        buffer_kwargs (dict | None): ``ReplayBuffer`` controls ``maxlen``,
            ``sample_num``, ``insert_num``, ``seed``, and the optional storage
            ``strategy``. Omitting ``strategy`` retains FIFO exactly.
        baseline (str | None): Optional named research control. ``None`` leaves
            every legacy switch independent; named controls validate a coherent
            sequential, cumulative, reservoir-ER, LwF, generative-replay, or
            joint-DiT treatment.
        plot_results (bool): Plot accuracy against the number of seen classes.
        generative_model_kwargs (dict): Replay-model controls ``train_num``
            and ``samples_per_class``.
        use_generative_replay (bool): Generate old examples between tasks when
            a replay model is present. The default ``True`` preserves previous
            behavior; ``False`` enables joint/KD controls without generation.
        replay_budget_mode (str): ``"legacy"`` retains per-class generation and
            buffer counts. ``"fixed_total"`` uses the explicit old/current
            example budgets below so replay methods receive matched exposure.
        replay_old_examples (int | None): Total old examples selected per task
            in fixed-total mode.
        replay_current_examples (int | None): Optional exact current-data
            exposure per task in fixed-total mode; ``None`` keeps every current
            example.
        optimizer_steps_per_epoch (int | None): Optional positive optimizer
            update count for each active training phase per epoch. ``None``
            preserves the existing finite-dataset fit behavior.
        replay_candidate_multiplier (int): Positive candidate-pool multiplier
            used before an optional cognitive replay gate.
        replay_selection (str): ``"all"`` legacy selection or the optional
            ``"uniform"``, ``"random"`, ``"confidence"``, ``"surprise"``,
            or ``"confidence_surprise"`` matched-pool controls.
        replay_surprise_weight (float): Combined-gate surprise weight in
            ``[0, 1]``.
        replay_cache_dir (str | None): Optional shared candidate-pool directory
            for matched raw-versus-EMA or gate comparisons.
        replay_cache_mode (str): ``"off"``, ``"write"``, ``"read"``, or
            ``"read_write"``. The default performs no cache I/O.
        mechanistic_metrics (bool): Compute optional teacher calibration and
            replay consistency/coverage/diversity/drift outcomes per task.
        mechanistic_max_samples (int): Positive cap on quadratic diversity work.
        use_generative_model_classifier (bool): Use the classifier attached to
            a diffusion replay model instead of the standalone model.
        train_classifier_separately (bool): Add the separate classifier phase
            required by ``DiffusionClassifierV2``. Standard diffusion
            classifiers train both parts jointly.
        use_distillation (bool): Use each completed diffusion-classifier raw
            student as the next continual task's frozen teacher. The model must
            provide a distillation token and a positive teacher objective. An
            optional runtime teacher may initialize task one but is never
            stored in this YAML-safe section.
        snapshot_network_name (str): ``"raw"`` or ``"ema"`` branch cloned for
            previous-task distillation.
        use_ensemble_accuracy (bool): Use timestep-ensemble values as the
            authoritative continual accuracy matrix and derived metrics.
        evaluate_ensemble_accuracy (bool): Also evaluate diffusion-classifier
            ensemble accuracy after each continual task. Retained as a legacy
            alias for ``use_ensemble_accuracy``.
        ensemble_accuracy_kwargs (dict): Options forwarded to
            ``DiffusionClassifier.evaluate_ensemble_accuracy``.
        return_details (bool): Return task histories and final models together
            with accuracies from a direct configured call.
        save_task_checkpoints (bool): Persist one atomic recovery checkpoint
            after every completed task.
        checkpoint_dir (str | None): Optional recovery root. Configured runs
            default to ``<results_path>/checkpoints``.
        resume_from (str | None): Checkpoint root or committed task directory
            from which the next unfinished task is restored.
        experiment_phase (str): ``"legacy"`` preserves test reporting,
            ``"development"`` prohibits test evaluation, and
            ``"confirmation"`` enables the frozen confirmatory run path.
        experiment_manifest_path (str | None): Frozen paired-block manifest
            required for confirmation runs.
        experiment_manifest_hash (str | None): Trusted external SHA-256 digest
            used to authenticate the confirmation manifest.
        experiment_run_id (str | None): Planned condition-by-stream run whose
            schedule and seed this invocation must match.
    """

    class_num: int | None = None
    class_order: list[int] | None = None
    task_groups: list[list[int]] | None = None
    task_size: int = 1
    class_order_mode: str = "fixed"
    task_order_mode: str = "fixed"
    seed: int | None = None
    remove_prev_classes: bool = True
    keep_same_model: bool = True
    use_loaded_opt: bool = True
    use_buffer: bool = False
    buffer_kwargs: dict[str, object] | None = field(
        default_factory=_default_buffer_kwargs
    )
    baseline: str | None = None
    plot_results: bool = True
    generative_model_kwargs: dict[str, int] = field(
        default_factory=_default_generative_replay_kwargs
    )
    use_generative_replay: bool = True
    replay_budget_mode: str = "legacy"
    replay_old_examples: int | None = None
    replay_current_examples: int | None = None
    replay_candidate_multiplier: int = 1
    replay_selection: str = "all"
    replay_surprise_weight: float = 0.5
    replay_cache_dir: str | None = None
    replay_cache_mode: str = "off"
    mechanistic_metrics: bool = False
    mechanistic_max_samples: int = 512
    use_generative_model_classifier: bool = False
    train_classifier_separately: bool = False
    use_distillation: bool = False
    snapshot_network_name: str = "raw"
    use_ensemble_accuracy: bool = False
    evaluate_ensemble_accuracy: bool = False
    ensemble_accuracy_kwargs: dict[str, object] = field(default_factory=dict)
    return_details: bool = False
    save_task_checkpoints: bool = False
    checkpoint_dir: str | None = None
    resume_from: str | None = None
    experiment_phase: str = "legacy"
    experiment_manifest_path: str | None = None
    experiment_manifest_hash: str | None = None
    experiment_run_id: str | None = None
    optimizer_steps_per_epoch: int | None = None

@dataclass
class TrainingConfig:
    """Task, fit-loop, callback, TensorBoard, and persistence settings.

    Attributes:
        project_tag (str | None): Optional result-run identifier passed to the
            image callback; ``None`` lets that callback choose one.
        epochs (int): Positive fit epoch count; defaults to 20.
        fit_method (str): Select ``"fit"`` or ``"fit_progressively"``.
            Progressive training uses the curriculum fields below instead of
            forwarding ``epochs`` to the wrapper.
        stage_tasks (list[object] | str | None): Ordered progressive stage
            descriptions, a ``"timesteps_only"``, ``"resolutions_only"``, or
            ``"depths_only"`` shorthand, or ``None`` outside progressive mode.
            YAML lists may represent the method's tuple-like descriptions.
        stages_num (int | None): Optional count used to generate shorthand
            progressive stages when companion values do not determine it.
        stages_verbose (bool): Print each resolved progressive stage.
        stage_epochs (int): Maximum epochs allocated to every listed stage.
        final_epochs (int | None): Epochs for the final full-task stage;
            ``None`` uses ``stage_epochs`` and ``0`` disables that stage.
        timestep_boundaries (list[list[int] | None] | None): Optional
            stage-indexed lower/upper timestep pairs.
        timestep_clustering_type (str): ``"uniform"`` or ``"log_snr"``
            clustering used when timestep boundaries are generated.
        resolutions (list[int | None] | None): Optional stage-indexed square
            input resolutions.
        depths (list[object | None] | None): Optional stage-indexed network
            depth-growth specifications.
        pacing_type (str): ``"fixed"`` or ``"plateau"`` stage pacing.
        earlystopping_type (str): ``"batch_wise"`` or ``"epoch_wise"``
            stopping used by plateau pacing.
        progressive_monitor (str): Metric forwarded as ``monitor`` to
            progressive plateau stopping.
        progressive_patience (int): Value forwarded as ``patience`` to
            progressive plateau stopping.
        min_delta (float): Minimum progressive plateau improvement.
        stopper_mode (str): Keras mode for epoch-wise progressive stopping.
        fit_kwargs (dict[str, object]): Additional Keras fit arguments such as
            step counts; each config instance owns an independent mapping.
        use_valset (bool): Build and pass validation data for the selected
            dataset when the loader created an explicit split. Loaders without
            a split leave validation disabled; test rows are never substituted.
        show_images (bool): Display callback sample grids during training.
        save_gifs (bool): Save callback denoising animations during training.
            Trajectory callbacks are skipped for no-EMA and VAE/swap models.
        results_path (str | os.PathLike[str] | None): Base artifact directory
            passed to the image callback. ``None`` is supported only by
            display-only runs whose runtime saving options are all disabled.
        save_weights (bool): Save final wrapper weights and record their path.
            Dynamic diffusion weights require a paired updated config file;
            training writes it even if ordinary config saving was disabled.
        task (str): ``legacy``, ``generation``, ``joint``, ``classification``,
            or ``continual``.
        seed (int | None): TensorFlow/Keras and dataset split seed.
        dtype_policy (str): Keras global dtype policy installed before data and
            model construction, such as ``"float32"``, ``"mixed_float16"`` or
            ``"mixed_bfloat16"``.
        deterministic_ops (bool): Request deterministic TensorFlow kernels when
            supported. Continual runs still derive every random source from
            ``continually_learn.seed``.
        verbose (int): Keras and project reporting verbosity.
        patience (int): Early-stopping patience; ``0`` disables it.
        monitor (str | None): Explicit early-stopping metric, or ``None`` for
            the training API default.
        monitor_mode (str): ``"auto"``, ``"min"``, or ``"max"``.
        tensorboard (bool): Write TensorBoard summaries when true.
        tensorboard_path (str | None): Optional TensorBoard root.
        tensorboard_run_name (str | None): Event-file suffix used by HPO to
            encode every sampled value.
        report_every_epoch (bool): Enable compatible diffusion image callbacks.
    """

    project_tag: str | None = None
    epochs: int = 20
    fit_method: str = "fit"
    stage_tasks: list[object] | str | None = None
    stages_num: int | None = None
    stages_verbose: bool = True
    stage_epochs: int = 1
    final_epochs: int | None = None
    timestep_boundaries: list[list[int] | None] | None = None
    timestep_clustering_type: str = "log_snr"
    resolutions: list[int | None] | None = None
    depths: list[object | None] | None = None
    pacing_type: str = "fixed"
    earlystopping_type: str = "epoch_wise"
    progressive_monitor: str = "val_noise_loss"
    progressive_patience: int = 10
    min_delta: float = 1e-3
    stopper_mode: str = "min"
    fit_kwargs: dict[str, object] = field(default_factory=dict)
    use_valset: bool = True
    show_images: bool = False
    save_gifs: bool = True
    results_path: str | os.PathLike[str] | None = "./results"
    save_weights: bool = True
    task: str = "legacy"
    seed: int | None = None
    dtype_policy: str = "float32"
    deterministic_ops: bool = False
    verbose: int = 1
    patience: int = 0
    monitor: str | None = None
    monitor_mode: str = "auto"
    tensorboard: bool = False
    tensorboard_path: str | None = None
    tensorboard_run_name: str | None = None
    report_every_epoch: bool = True

    def __post_init__(self: TrainingConfig) -> None:
        """Convert an optional path-like artifact root.

        Returns:
            None: The path is normalized in place when present.
        """

        # Leave disabled artifact output unchanged.
        if self.results_path is not None:
            self.results_path = os.fspath(self.results_path)


@dataclass
class ReportingConfig:
    """Post-training evaluation, plot, sample, GIF, and CSV controls.

    Attributes:
        show_history_plot (bool): Display history figures interactively.
        save_history_plot (bool): Save history figures in the result directory.
        final_images_cfg_scale (float): CFG scale for final generation.
        final_images_steps (int): Reverse-diffusion steps for final samples;
            it must satisfy wrapper sampling bounds.
        show_final_images (bool): Display the final sample grid.
        save_final_images (bool): Save the final sample grid as PNG.
        save_final_gifs (bool): Request sampling trajectories and save a GIF;
            VAE/swap sampling has no trajectories, so only images are reported.
        plot_without_20percent (bool): Also plot history after discarding the
            first ``int(0.2 * epochs)`` epochs.
        run_trainset_eval (bool): Evaluate EMA and raw networks on training data.
        run_valset_eval (bool): Evaluate EMA and raw networks on validation
            data; requires a non-``None`` validation dataset.
        evaluate_ensemble_accuracy (bool): Also evaluate ensemble accuracy for
            ``DiffusionClassifier`` and ``DiffusionClassifierV2`` models.
        ensemble_accuracy_kwargs (dict): Options forwarded to
            ``DiffusionClassifier.evaluate_ensemble_accuracy``. Reporting
            selects the raw or EMA network for each evaluation.
        save_csv (bool): Save epoch metrics and enabled evaluations as CSV.
    """

    show_history_plot: bool = False
    save_history_plot: bool = True
    final_images_cfg_scale: float = 3.0
    final_images_steps: int = 1_000
    show_final_images: bool = False
    save_final_images: bool = True
    save_final_gifs: bool = True
    plot_without_20percent: bool = True
    run_trainset_eval: bool = True
    run_valset_eval: bool = True
    evaluate_ensemble_accuracy: bool = False
    ensemble_accuracy_kwargs: dict[str, object] = field(default_factory=dict)
    save_csv: bool = True


@dataclass
class Config:
    """Root configuration object consumed by the training entry point.

    Attributes:
        dataset (DatasetConfig): Input batching and shuffling settings.
        model (ModelConfig): Network/wrapper selection and configuration.
        optimizer (OptimizerConfig): Adam cosine-decay settings.
        training (TrainingConfig): Fit and artifact settings.
        continually_learn (ContinuallyLearnConfig): Class-incremental loop and
            replay settings.
        reporting (ReportingConfig): Post-training reporting settings.
        hpo (dict): Resolved study/trial metadata, sampled values, and selected
            accuracy feedback signal.

    All fields accept either their declared dataclass instance or a mapping in
    ``Config(...)``/YAML input.  Missing top-level or nested fields use
    dataclass defaults; unknown keys raise ``TypeError`` during construction.
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    continually_learn: ContinuallyLearnConfig = field(
        default_factory=ContinuallyLearnConfig
    )
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    hpo: dict = field(default_factory=dict)

    def __post_init__(self: Config) -> None:
        """Convert every supplied top-level mapping to its section dataclass.

        Returns:
            None: Section attributes are normalized in place.
        """

        self.dataset = _section(self.dataset, DatasetConfig)
        self.model = _section(self.model, ModelConfig)
        self.optimizer = _section(self.optimizer, OptimizerConfig)
        self.training = _section(self.training, TrainingConfig)
        self.continually_learn = _section(
            self.continually_learn, 
            ContinuallyLearnConfig
        )
        self.reporting = _section(self.reporting, ReportingConfig)


def _section(
    value: SectionT | Mapping[str, object], 
    section_type: type[SectionT]
) -> SectionT:
    """Return an existing config section or construct one from a mapping.

    Args:
        value (object | Mapping[str, object]): Existing ``section_type``
            instance or keyword mapping.  ``None`` is not treated as an omitted
            section; it fails when expanded.
        section_type (type): Dataclass class to recognize/construct.

    Returns:
        object: ``value`` unchanged when already of ``section_type``;
        otherwise ``section_type(**value)``.

    Raises:
        TypeError: If ``value`` is not a compatible mapping or has unknown keys.
    """

    # Reuse an already constructed typed section unchanged.
    if isinstance(value, section_type):
        return value

    return section_type(**value)


def load_config(
    path: str | os.PathLike[str] | None = None
) -> Config:
    """Load a complete config tree from YAML or dataclass defaults.

    Args:
        path (str | os.PathLike | None): YAML file path.  ``None`` returns
            ``Config()`` and does not read ``configs/default.yaml``.  A YAML
            file may omit sections/fields to receive dataclass defaults, but
            its root must be a mapping.

    Returns:
        Config: Typed root with all nested mappings converted to section
        dataclasses.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        yaml.YAMLError: If YAML syntax is invalid or a mapping repeats an
            explicit key.
        TypeError: If a YAML section is not a mapping or contains an unknown
            dataclass field.
    """

    # Return pure dataclass defaults when no YAML path is supplied.
    if path is None:
        return Config()

    with open(path, "r", encoding="utf-8") as stream:
        data = _safe_load_unique_yaml(stream)
    
    return Config(**data)


def _declared_default(config_field: Field[Any]) -> tuple[bool, object]:
    """Return a fresh declared default for one dataclass field.

    Args:
        config_field (Field[Any]): Field whose ``default`` or
            ``default_factory`` should be inspected.

    Returns:
        tuple[bool, object]: Whether a default exists and its value. Factories
        are invoked for each comparison so mutable defaults remain isolated.
    """

    # Reuse immutable and explicitly declared field defaults directly.
    if config_field.default is not MISSING:
        return True, config_field.default

    # Create a fresh list, mapping, or nested config for factory defaults.
    if config_field.default_factory is not MISSING:
        return True, config_field.default_factory()

    return False, None


def _equals_default(value: object, default: object) -> bool:
    """Return whether a serializable field value equals its declared default.

    Args:
        value (object): Current field value.
        default (object): Fresh declared default value.

    Returns:
        bool: ``True`` only when equality produces one unambiguous truth value.
    """

    try:
        return bool(value == default)
    except (TypeError, ValueError):
        # Preserve unusual values whose equality cannot be reduced to one bool.
        return False


def _shortened_dataclass(
    value: object, 
    serialized: Mapping[str, object], 
    baseline: object = MISSING, 
) -> dict[str, object]:
    """Return changed dataclass fields while preserving their nested shape.

    Args:
        value (object): Dataclass instance being compared.
        serialized (Mapping[str, object]): Its recursively converted ``asdict``
            representation, used to retain normal deep-copy behavior.
        baseline (object): Optional parent-field default instance. ``MISSING``
            compares fields with their own declarations.

    Returns:
        dict[str, object]: Recursive mapping containing only nondefault fields.
    """

    shortened: dict[str, object] = {}
    for config_field in fields(value):
        current_value = getattr(value, config_field.name)

        # Nested baselines preserve defaults supplied by a parent factory.
        if baseline is not MISSING:
            has_default = True
            default_value = getattr(baseline, config_field.name)
        # Fall back to this dataclass type's field declarations at the root.
        else:
            has_default, default_value = _declared_default(config_field)

        current_is_dataclass = (
            is_dataclass(current_value) and not isinstance(current_value, type)
        )
        default_is_dataclass = (
            has_default
            and is_dataclass(default_value)
            and not isinstance(default_value, type)
        )

        # Recurse so one changed leaf retains all required section keys.
        if current_is_dataclass:
            child_baseline = default_value if default_is_dataclass else MISSING
            child_serialized = serialized[config_field.name]
            child_shortened = _shortened_dataclass(
                current_value,
                child_serialized,
                child_baseline,
            )
            # Omit an unchanged optional nested section.
            if has_default and not child_shortened:
                continue
            shortened[config_field.name] = child_shortened
            continue

        # Omit scalar and container values equal to fresh declared defaults.
        if has_default and _equals_default(current_value, default_value):
            continue
        shortened[config_field.name] = serialized[config_field.name]

    return shortened


def save_config(
    config: Config, 
    config_path: str | os.PathLike[str],
    shorten: bool = False
) -> None:
    """Serialize a full or default-pruned config dataclass tree to YAML.

    Args:
        config (Config): Dataclass root to convert recursively with ``asdict``.
        config_path (str | os.PathLike): Destination file.  Existing content is
            overwritten; parent directories must already exist.
        shorten (bool): When ``False`` (the default), save every field exactly
            as before. When ``True``, recursively omit values equal to their
            declared dataclass defaults. Nonempty plain mappings such as
            ``hpo`` remain intact because their entries have no declared field
            defaults.

    Returns:
        None.

    Raises:
        TypeError: If ``config`` is not a dataclass instance or contains values
            PyYAML cannot represent.
        OSError: If the destination cannot be opened or written.
    """

    # Preserve the original open, conversion, and dump order for full saves.
    if not shorten:
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(asdict(config), file, sort_keys=True)
        return

    serialized = asdict(config)
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            _shortened_dataclass(config, serialized),
            file,
            sort_keys=True,
        )
