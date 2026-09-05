"""Typed experiment configuration, schedule resolution, and safe YAML persistence.

Config owns dataset, model/wrapper, optimizer, training, continual-learning,
reporting, and HPO metadata sections. Typed architecture dataclasses describe
raw constructor settings; ModelConfig also supports explicit family names and
generic keyword mappings for the shared model factory. Creating configuration
objects does not train models or allocate their TensorFlow variables.

load_config normalizes nested mappings and rejects explicit duplicate YAML
keys. save_config writes every field or a compact tree that omits declared
defaults while preserving reload behavior. resolve_continual_schedule validates
original-label orders/groups and optionally shuffles them with a local seed.
Mutable defaults are created independently for every dataclass instance.
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
        visited (set[int] | None): Identity set used to terminate recursive YAML alias
            graphs. Callers normally omit it. Defaults to ``None``.

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
        task (object): Candidate task name, expected to be a string despite the broad object
            annotation. Supported values are 'legacy', 'generation', 'joint',
            'classification', and 'continual'; case is ignored, but surrounding whitespace
            is not stripped.

    Returns:
        str: Lowercase supported task selector.

    Raises:
        ValueError: If the normalized task is unsupported.
        AttributeError: If task does not provide the expected string lower() method.
    """

    normalized = task.lower()
    # Reject unknown tasks before dataset/model construction can misroute them.
    if normalized not in _TRAINING_TASKS:
        raise ValueError(
            "training task must be one of " + str(sorted(_TRAINING_TASKS)) + "."
        )

    return normalized


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
        class_order (Sequence[int] | None): Unique original dataset labels. None derives
            their order from explicit task_groups or natural range(class_num) when groups
            are absent. Defaults to ``None``.
        task_groups (Sequence[Sequence[int]] | None): Nonempty groups introduced at
            successive tasks. None creates contiguous groups of task_size from the class
            order; explicit groups require class_order_mode='fixed'. Defaults to ``None``.
        available_class_num (int | None): Optional dataset class count used to validate
            label bounds and fill a completely unspecified schedule. Defaults to
            ``None``.
        task_size (int): Positive number of classes placed in each automatically
            constructed task. The final task may contain fewer classes. Defaults to
            ``1``.
        class_order_mode (str): ``"fixed"`` preserves the supplied/natural class order;
            ``"random"`` shuffles it reproducibly before grouping. Defaults to
            ``'fixed'``.
        task_order_mode (str): ``"fixed"`` preserves task order; ``"random"`` shuffles
            complete groups without changing within-task class order. An automatic short
            remainder is never placed first. Defaults to ``'fixed'``.
        seed (int | None): Local random.Random seed for optional class/task shuffling. None
            uses Python's default random initialization and does not promise repeatable
            random ordering; fixed modes need no random draws. Defaults to ``None``.

    Returns:
        tuple[list[int], list[list[int]]]: New resolved original-label order and
        groups whose flattening equals that order. Supplied containers are not
        mutated; optional shuffling is applied only to the returned lists.

    Raises:
        ValueError: If the schedule is empty, duplicated, inconsistent, out of
            range, or uses an unsupported ordering mode.
    """

    class_order_mode = str(class_order_mode).lower()
    task_order_mode = str(task_order_mode).lower()
    # Keep an omitted seed unset; normalize an explicit seed to an integer.
    seed = None if seed is None else int(seed)
    task_size = int(task_size)
    # Reject task widths that cannot produce nonempty class groups.
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
    # Flatten explicit groups; leave order inference open when groups are absent.
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
        # Use the dataset class count only when no selected count was supplied.
        resolved_count = available_class_num if class_num is None else class_num
        # Require some source of truth for the number of classes to schedule.
        if resolved_count is None:
            raise ValueError(
                "class_num, class_order, task_groups, or available_class_num "
                "must define the continual schedule."
            )
        resolved_order = list(range(resolved_count))

    resolved_order = [int(label) for label in resolved_order]
    # Normalize class IDs within explicitly supplied groups as well.
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

    # Reject a schedule with no transition between separate tasks.
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
        # Reinsert a short remainder after the first full task.
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
    """Expose a dataclass configuration as independent constructor keywords.

    Subclasses must be dataclasses: kwargs() uses dataclasses.asdict, recursively
    converts nested dataclasses, and copies mutable containers. No network is
    constructed and no default-valued or None-valued field is omitted.
    """

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
    """Arguments forwarded to ``DiffusionTransformer``.

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory.

    Attributes:
        num_classes (int | None): Positive number of real classes. ``None`` starts with no
            real classes and enables class-by-class growth; this mode requires
            classifier-free guidance. Defaults to ``10``.
        use_cfg (bool): Whether the label vocabulary includes null label 0 for
            classifier-free guidance. The wrapper shifts real labels by one when this is
            true. Defaults to ``True``.
        timesteps (int): Size of the discrete time-embedding vocabulary. Defaults to
            ``1000``.
        image_size (int): Base square image size. It must be divisible by ``patch_size``.
            Defaults to ``28``.
        channels (int): Number of image and output channels. Defaults to ``1``.
        patch_size (int): Side length of each non-overlapping patch. Defaults to ``2``.
        dim (int): Depth-0 patch feature width. Defaults to ``32``.
        dim_forced (bool): If true, connectors and spatial layers project feature growth
            back toward the inferred base width when needed. Defaults to ``True``.
        patchify_with_cnn (bool): Use a two-convolution ``same``-padding stem instead of the
            single patch-size/stride projection. Defaults to ``False``.
        patches_pos_embed_type (str): Patch positional encoding: ``"new_weight"``,
            ``"1d_sincos"``, ``"1d_interpolate"``, ``"1d_learned_interpolate"``,
            ``"2d_sincos"``, ``"2d_interpolate"``, or ``"2d_learned_interpolate"``. Defaults
            to ``'2d_sincos'``.
        patches_pos_merger_type (str): ``"add"`` preserves width; ``"concat"`` appends
            positional channels. Defaults to ``'add'``.
        patches_conds_merger_type (str | None): How to inject the repeated condition into
            every patch. ``None`` leaves patches separate, ``"add"`` requires ``dim ==
            cond_dim``, and ``"concat"`` increases the token width. Defaults to ``None``.
        shift_inputs (bool): Prepend the patch embedder's learned BOS token and drop the
            final patch, providing right-shifted decoder input. Defaults to ``False``.
        cond_dim (int | None): Combined conditioning width. ``None`` uses ``dim``. With
            ``conds_merger_type="concat"`` and both time and label conditions, each
            individual embedder uses half this width. Defaults to ``None``.
        cond_type (str | None): Adaptive-normalization/patch condition: ``"time_label"``,
            ``"time"``, ``"label"``, or ``None``. ``None`` requires
            ``ln_no_adaptation=True``. Defaults to ``'time_label'``.
        conds_merger_type (str): Combine simultaneous time and label embeddings by ``"add"``
            or ``"concat"``. Defaults to ``'add'``.
        time_embed_type (str): Encoding used for integer timesteps. Use ``"new_weight"`` or
            ``"1d_sincos"``; spatial/interpolation modes do not produce the rank-2 table
            required by condition lookup. Defaults to ``'1d_sincos'``.
        time_freq_dim (int | None): Optional pre-MLP time embedding width; ``None`` uses the
            condition-embedder width. Defaults to ``None``.
        time_embed_trainable (bool): Whether a non-``"new_weight"`` table initialized from
            the selected time encoding may be trained. ``"new_weight"`` is always trainable.
            Defaults to ``False``.
        time_mlp_ratio (float | None): Hidden-width ratio for the time-embedding projection.
            None omits its hidden layer when time_freq_dim is None; an explicit
            time_freq_dim makes an omitted ratio resolve to 1. Defaults to ``None``.
        label_embed_type (str): Encoding/table used for label IDs. ``"new_weight"`` and
            ``"1d_sincos"`` are the valid rank-2 lookup-table modes. Defaults to
            ``'new_weight'``.
        label_embed_trainable (bool): Whether a non-learned initialized label table can
            train; ``"new_weight"`` is always trainable. Defaults to ``False``.
        label_freq_dim (int | None): Label lookup-table width before projection; None uses
            the resolved label-condition width. An explicit width enables a projection back
            to that condition width. Defaults to ``None``.
        label_mlp_ratio (float | None): Hidden-width ratio for the label-embedding
            projection. None omits its hidden layer when label_freq_dim is None; an explicit
            label_freq_dim makes an omitted ratio resolve to 1. Defaults to ``None``.
        cls_token_type (str | None): ``"new_weight"`` creates a learned token; ``"time"``,
            ``"label"``, or ``"time_label"`` derives it from those conditions; ``None`` adds
            no token. Defaults to ``None``.
        cls_token_freq_dim (int | None): Class-token component width before projection. None
            uses the token width, or half that width when concatenating a positional
            component. An explicit width enables projection back to the component width;
            unused if cls_token_type is None. Defaults to ``None``.
        cls_token_mlp_ratio (float | None): Hidden-width ratio for the class-token
            projection. None omits the hidden layer when cls_token_freq_dim is None;
            supplying a frequency width makes an omitted ratio resolve to 1.
            Condition-derived tokens may still need an output projection. Defaults to
            ``None``.
        cls_token_pos_merger_type (str): ``"add"`` or ``"concat"`` for the token's
            learned/positional representation. Defaults to ``'add'``.
        distil_token_type (str | None): Distillation-token source, with the same
            ``"new_weight"``, ``"time"``, ``"label"``, ``"time_label"``, and ``None``
            choices as ``cls_token_type``. Defaults to ``None``.
        distil_token_freq_dim (int | None): Distillation-token component width before
            projection, with the same None/inferred-width behavior as cls_token_freq_dim.
            Unused if distil_token_type is None. Defaults to ``None``.
        distil_token_mlp_ratio (float | None): Hidden-width ratio for the distillation-token
            projection, with the same None and explicit-frequency-width rules as
            cls_token_mlp_ratio. Unused when no distillation token is present. Defaults to
            ``None``.
        distil_token_pos_merger_type (str): Positional merge for the distillation token,
            matching ``cls_token_pos_merger_type``. Defaults to ``'add'``.
        depth (int): Number N of processing stages. ``0`` creates only depth-0 embedding
            plus the output head; ``1..N`` create those many ordered dictionaries in
            ``layers_dicts``. Defaults to ``2``.
        connection_ids_dict (dict): Maps a target depth to feature depths combined before
            that stage. Keys are in ``1..depth``; source IDs must precede the target after
            normalization. Example: ``{2: [0, 1]}`` concatenates depth 0 and depth 1 at
            stage 2. Defaults to a fresh ``{}`` for each instance.
        connection_kwargs (dict): Options shared by every feature connector. Allowed keys
            are ``connect_axis`` (int), ``connect_type`` (``"concat"`` or ``"add"``),
            ``use_layer_norm`` (bool), ``ln_dim`` (int | None), ``ln_mlp_ratio`` (float |
            None), ``ln_no_adaptation`` (bool), ``mlp_output_dim`` (int | None),
            ``mlp_ratio`` (float | None), and ``mlp_activation_func`` (Keras activation
            name/callable). For ``"add"``, all selected tensors must have identical shape.
            Defaults to a fresh ``{}`` for each instance.
        cross_attention_ids_dict (dict): Same ID syntax as ``connection_ids_dict``, but
            constructs the tensor plugged into attention as external queries or values.
            Defaults to a fresh ``{}`` for each instance.
        cross_attention_kwargs (dict): Same allowed keys and value rules as
            ``connection_kwargs``. Defaults to a fresh ``{}`` for each instance.
        cross_attention_plug_type (str): External cross-attention side: 'values' supplies
            keys/values to queries from the main stream; 'queries' supplies external queries
            to keys/values from the main stream. Defaults to ``'values'``.
        vit_block_ids (list[int | None]): Stage IDs containing attention blocks. ``[None]``
            means every depth, ``[]`` means none, and negative IDs are relative to the end.
            Defaults to a fresh ``[None]`` for each instance.
        use_decoder_ids (list[int | None]): Subset of block IDs implemented by
            ``DiTDecoderBlock``; all other block IDs use encoder blocks. Defaults to a fresh
            ``[]`` for each instance.
        mha_key_dim (int | None): Per-head key width; ``None`` lets the attention layer
            infer it. Defaults to ``None``.
        mha_value_dim (int | None): Per-head value width; ``None`` uses the attention
            implementation's default. Defaults to ``None``.
        mha_num_heads (int): Number of attention heads. Defaults to ``4``.
        vit_block_mlp_ratio (float): Transformer FFN hidden expansion. Defaults to ``4.0``.
        vit_block_mlp_output_dims (dict[int, int]): Optional per-depth FFN output widths,
            for example ``{3: 128}``. Defaults to a fresh ``{}`` for each instance.
        ln_mlp_ratio (float | None): Hidden-width ratio for adaptive layer-normalization
            projections. None omits the optional hidden expansion; ln_no_adaptation=True
            disables condition adaptation entirely. Defaults to ``None``.
        ln_no_adaptation (bool): Use ordinary layer normalization without a
            condition-dependent affine/gate path. Defaults to ``False``.
        drop_prob (float): Transformer residual-drop probability in ``[0, 1]``. Defaults to
            ``0.0``.
        drop_per_sample (bool): Apply residual dropping independently per sample instead of
            sharing a decision across the batch. Defaults to ``True``.
        local_mixer_ids (list[int | None]): Depths with a depthwise spatial token mixer. ID
            handling matches ``vit_block_ids``. Defaults to a fresh ``[]`` for each
            instance.
        local_mixer_kwargs (dict): Shared mixer overrides. Allowed keys are
            ``embed_temperature`` (float), ``dim`` (int), ``grid_size`` (int),
            ``use_layer_norm`` (bool), ``ln_mlp_ratio`` (float | None), ``ln_no_adaptation``
            (bool), ``kernel_size`` (int), ``strides`` (int), ``depth_multiplier`` (int),
            ``use_pointwise`` (bool), ``pointwise_dim_ratio`` (int), ``zero_init`` (bool),
            ``pos_embed_type`` (a positional type or ``None``), ``pos_interpolation_method``
            (``tf.image.resize`` method), ``pos_merger_type`` (``"add"``/``"concat"``), and
            ``mlp_ratio``, ``mlp_activation_func``, ``mlp_output_dim``. Defaults to a fresh
            ``{}`` for each instance.
        downsample_ids (list[int | None]): Depths that reduce each spatial grid dimension,
            normally by two. Defaults to a fresh ``[]`` for each instance.
        downsample_kwargs (dict): Allowed keys are ``embed_temperature``, ``dim``,
            ``grid_size``, ``use_layer_norm``, ``ln_mlp_ratio``, ``ln_no_adaptation``,
            ``scaling_method`` (``"avg_pooling"``, ``"max_pooling"``, or ``"cnn_stride"``),
            ``cnn_dim_ratio`` (int), ``cnn_kernel_size`` (int), ``cnn_activation_func``
            (Keras activation), positional ``pos_embed_type``, ``pos_interpolation_method``,
            ``pos_merger_type``, and MLP ``mlp_ratio``, ``mlp_activation_func``,
            ``mlp_output_dim``. Defaults to a fresh ``{}`` for each instance.
        upsample_ids (list[int | None]): Depths that double each spatial grid dimension.
            Defaults to a fresh ``[]`` for each instance.
        upsample_kwargs (dict): Same common embedding, layer norm, position, and MLP keys as
            ``downsample_kwargs`` plus ``scaling_method`` (``"cnn_transpose"``,
            ``"interpolate"``, or ``"cnn_interpolate"``), ``scaling_interpolation_method``
            (Keras ``UpSampling2D`` interpolation), ``cnn_dim_ratio``, ``cnn_kernel_size``,
            and ``cnn_activation_func``. Defaults to a fresh ``{}`` for each instance.
        reshaper_ids_dict (dict[int, str]): Maps depths to ``"flatten"`` (tokens to one
            vector) or ``"unflatten"`` (vector back to the inferred token grid). Reshapers
            form consecutive pairs, for example ``{2: "flatten", 3: "unflatten"}``. Defaults
            to a fresh ``{}`` for each instance.
        reshaper_kwargs (dict): Only ``add_kl`` (bool) and ``latent_dim_ratio`` (a list of
            positive floats) are allowed. The list has one entry per flatten/unflatten pair
            in ascending flatten-depth order. With ``add_kl=True``, each flatten reshaper
            returns a sampled latent, mean, and log variance for a VAE KL objective.
            Defaults to a fresh ``{}`` for each instance.
        cls_token_regularizer_ids (list[int | None]): Depth IDs whose token slice feeds an
            auxiliary ``num_classes`` softmax. ID 0 applies a regularizer to the label
            embedding; ``[None]`` selects 0..N. Defaults to a fresh ``[]`` for each
            instance.
        cls_token_regularizer_kwargs (dict): ``start`` and ``end`` are Python token-slice
            bounds. Optional ``mlp_ratio`` adds a hidden Dense layer, and
            ``activation_function`` selects its activation. Missing values default to
            ``None`` and ``"tanh"``, respectively. ``train_type`` is ``"normal"``,
            ``"distil"``, or ``"both"``; ``distil_type`` is ``"hard"`` or ``"soft"``.
            Defaults to a fresh ``{'start': 0, 'end': 1, 'train_type': 'normal',
            'distil_type': 'hard'}`` for each instance.
        final_ffn_activation_func (str): Activation on the zero-initialized patch-output
            projection. Defaults to ``'linear'``.
        use_refiner_cnn (bool): Add a two-convolution image-space refinement head after
            unpatchification. Defaults to ``False``.
        refiner_cnn_hidden_dim (int | None): Refiner hidden channels; ``None`` uses the
            current token width. Defaults to ``None``.
        refiner_cnn_residual (bool): Add the refiner output to the initial unpatchified
            image; false returns only the refinement. Defaults to ``True``.
        final_activation_func (str): Final output activation. Defaults to ``'linear'``.
        use_unpatchify (bool): Return image-shaped output when true; when false return final
            tokens of shape ``[B, tokens, features]``. Defaults to ``True``.
        name_prefix (str): Prefix inserted in generated layer names; an empty string adds no
            prefix. Defaults to ``''``.
        build (bool): Build symbolic inputs and variables when the raw network is
            constructed if True; False defers variable creation to an explicit build or
            first Keras call. Constructing this dataclass itself never builds variables.
            Defaults to ``True``.
    """

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
    vit_block_ids: list[int | None] = field(default_factory=lambda: [None])
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
    """Arguments forwarded to ``DiTClassifier``, including inherited ones.

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory. Inherited fields retain the meanings and
    defaults documented by :class:`DiffusionTransformerConfig`; the attributes below
    document this class's additions and explicit overrides.

    Attributes:
        aggregate_from_noises (bool): Classify the network's predicted noise image rather
            than selected internal main-branch features. This requires
            ``use_unpatchify=True``. Defaults to ``False``.
        feature_aggregation_ids_dict (dict): Maps classifier target depths to
            main-transformer feature IDs: 0 is embedded input and 1..depth are main stage
            outputs. Key 1 supplies the initial classifier feature and is required. A source
            -1 selects the final main depth; an item None expands all main depths. Each
            default instance starts with {1: [-1]}. Defaults to a fresh ``{1: [-1]}`` for
            each instance.
        feature_aggregation_kwargs (dict): Shared aggregation options. Accepted keys are
            ``connect_axis``, ``connect_type``, ``use_layer_norm``, ``ln_dim``,
            ``ln_mlp_ratio``, ``ln_no_adaptation``, ``mlp_output_dim``, ``mlp_ratio``, and
            ``mlp_activation_func``, with the same values and behavior as
            ``DiffusionTransformer.connection_kwargs``. Defaults to a fresh ``{}`` for each
            instance.
        cross_attention_aggregation_ids_dict (dict): Maps classifier target depths to main
            features used as external attention queries/values rather than the primary
            feature path. Defaults to a fresh ``{}`` for each instance.
        cross_attention_aggregation_kwargs (dict): Same exact allowed keys as
            ``feature_aggregation_kwargs``. Defaults to a fresh ``{}`` for each instance.
        classifier_only_cls_token (bool): Give the classifier its own prefix token and
            suppress the main branch's token. False lets both branches share the inherited
            main token behavior. Defaults to ``True``.
        classifier_only_distil_token (bool): Give the classifier its own distillation token
            and suppress the main branch's token. False lets both branches share
            ``distil_token_type``. Defaults to ``True``.
        clf_dim (int | None): Nominal classifier width. ``None`` is replaced after
            aggregation by ``first_aggregated_dim``. Defaults to ``None``.
        clf_dim_forced (bool): Project feature merges/spatial width changes back to
            ``clf_dim``. If true, ``clf_dim`` must be supplied. Defaults to ``False``.
        clf_cond_type (str | None): Classifier adaptive condition: ``"time_label"``
            (default), ``"time"``, ``"label"``, or None. This value does not inherit
            ``cond_type``. Defaults to ``'time_label'``.
        clf_cls_token_type (str | None): Classifier token source: ``"new_weight"``
            (default), ``"time_label"``, ``"time"``, ``"label"``, or None. It does not
            inherit ``cls_token_type``. Defaults to ``'new_weight'``.
        clf_distil_token_type (str | None): Classifier distillation-token source with the
            same choices as ``clf_cls_token_type``. ``None`` disables the classifier-only
            distillation token. It does not inherit ``distil_token_type``. Defaults to
            ``None``.
        clf_depth (int): Number of classifier processing depths. A separate terminal
            connector is created at clf_depth + 1. Defaults to ``1``.
        clf_connection_ids_dict (dict): Classifier target depth to earlier classifier source
            IDs. The mandatory key -1 names a terminal connector and is moved to clf_depth +
            1 during construction; its default source [-1] selects the final classifier
            processing depth. Example {2: [0, 1], -1: [-1]} also merges depths 0 and 1
            before stage 2. Defaults to a fresh ``{-1: [-1]}`` for each instance.
        clf_connection_kwargs (dict | None): Connector keys listed for
            ``feature_aggregation_kwargs``. ``None`` inherits the main branch's
            ``connection_kwargs``; ``{}`` requests layer defaults/inferred dimensions.
            Defaults to ``None``.
        clf_cross_attention_ids_dict (dict): Maps a classifier stage to earlier classifier
            features used for cross attention. Empty by default. Defaults to a fresh ``{}``
            for each instance.
        clf_cross_attention_kwargs (dict | None): Cross-attention connector options;
            ``None`` inherits the main branch. Defaults to ``None``.
        clf_cross_attention_plug_type (str | None): External-attention plug side. ``None``
            inherits the main ``cross_attention_plug_type`` (default ``"values"``). Defaults
            to ``None``.
        clf_vit_block_ids (list[int | None]): Classifier attention-block depths. ``[None]``
            means all 1..``clf_depth``; ``[]`` means no blocks. This list does not inherit
            from the main branch. Defaults to a fresh ``[None]`` for each instance.
        clf_use_decoder_ids (list[int | None]): Classifier block depths that use
            ``DiTDecoderBlock``. Empty by default. Defaults to a fresh ``[]`` for each
            instance.
        clf_mha_key_dim (int | None): Classifier per-head key width. It remains ``None`` by
            default and is inferred by the block. Defaults to ``None``.
        clf_mha_value_dim (int | None): Classifier per-head value width; remains ``None`` by
            default. Defaults to ``None``.
        clf_mha_num_heads (int | None): Head count; ``None`` inherits ``mha_num_heads`` (4
            by default). Defaults to ``None``.
        clf_vit_block_mlp_ratio (float | None): Classifier FFN expansion; ``None`` inherits
            ``vit_block_mlp_ratio`` (4 by default). Defaults to ``None``.
        clf_vit_block_mlp_output_dims (dict[int, int] | None): Optional classifier per-depth
            output widths; ``None`` copies the main mapping, while ``{}`` explicitly
            requests none. Defaults to ``None``.
        clf_ln_mlp_ratio (float | None): Classifier adaptive-normalization MLP ratio. This
            explicit default remains None; it does not inherit ``ln_mlp_ratio``. Defaults to
            ``None``.
        clf_ln_no_adaptation (bool | None): Disable condition adaptation; ``None`` inherits
            the main setting (false by default). Defaults to ``None``.
        clf_drop_prob (float | None): Residual-drop probability; ``None`` inherits the main
            value (0 by default). Defaults to ``None``.
        clf_drop_per_sample (bool | None): Drop residuals per sample; ``None`` inherits the
            main value (true by default). Defaults to ``None``.
        clf_local_mixer_ids (list[int | None]): Classifier local-mixer depths; empty by
            default and independent of main IDs. Defaults to a fresh ``[]`` for each
            instance.
        clf_local_mixer_kwargs (dict | None): Exact keys from ``local_mixer_kwargs``;
            ``None`` inherits the main mapping. Defaults to ``None``.
        clf_downsample_ids (list[int | None]): Classifier downsample depths. Defaults to a
            fresh ``[]`` for each instance.
        clf_downsample_kwargs (dict | None): Exact keys from ``downsample_kwargs``; ``None``
            inherits the main mapping. Defaults to ``None``.
        clf_upsample_ids (list[int | None]): Classifier upsample depths. Defaults to a fresh
            ``[]`` for each instance.
        clf_upsample_kwargs (dict | None): Exact keys from ``upsample_kwargs``; ``None``
            inherits the main mapping. Defaults to ``None``.
        clf_reshaper_ids_dict (dict[int, str]): Classifier depth to
            ``"flatten"``/``"unflatten"`` mapping; empty by default. Defaults to a fresh
            ``{}`` for each instance.
        clf_reshaper_kwargs (dict | None): Classifier variational-reshaper controls: add_kl
            (bool) and latent_dim_ratio (one positive float per consecutive
            flatten/unflatten pair in ascending flatten-depth order). None inherits a
            separate deep copy of the main reshaper_kwargs; an explicit mapping supplies
            independent classifier settings. Defaults to ``None``.
        clf_cls_token_regularizer_ids (list[int | None]): Classifier depths 0..``clf_depth``
            with auxiliary class softmax heads. Empty by default; ``[None]`` selects the
            full range. Defaults to a fresh ``[]`` for each instance.
        clf_cls_token_regularizer_kwargs (dict | None): Token slice and optional regularizer
            MLP settings. ``None`` inherits ``cls_token_regularizer_kwargs``. Missing
            ``mlp_ratio`` and ``activation_function`` values default to ``None`` and
            ``"tanh"``, respectively. Defaults to ``None``.
        force_global_avg_pooling (bool): Average all final tokens even when a class token is
            available. The distillation token is always excluded, but a class token remains
            in this average. Without a usable class token, global average pooling is
            selected. Defaults to ``False``.
        classifier_mlp_ratio (int | None): Add a hidden classifier Dense layer of
            ``final_width * ratio`` units when non-None. Defaults to ``None``.
        classifier_mlp_activation_func (str): Hidden classifier activation. Defaults to
            ``'tanh'``.
        dropout_rate (float): Classifier dropout rate; 0 omits dropout. Defaults to ``0.0``.
    """

    aggregate_from_noises: bool = False
    feature_aggregation_ids_dict: dict = field(
        default_factory=lambda: {1: [-1]}
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
        default_factory=lambda: {-1: [-1]}
    )
    clf_connection_kwargs: dict | None = None
    clf_cross_attention_ids_dict: dict = field(default_factory=dict)
    clf_cross_attention_kwargs: dict | None = None
    clf_cross_attention_plug_type: str | None = None
    clf_vit_block_ids: list[int | None] = field(default_factory=lambda: [None])
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
    """Arguments forwarded to ``DiTDecoder``.

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory. Inherited fields retain the meanings and
    defaults documented by :class:`DiffusionTransformerConfig`; the attributes below
    document this class's additions and explicit overrides.

    Attributes:
        encoder_output_grid_size (int | None): Final encoder token-grid side. None lets the
            standalone model factory infer image_size // patch_size. Composite
            encoder-decoder models derive this metadata from the actual encoder. Defaults to
            ``None``.
        encoder_output_dim (int | None): Final encoder feature width. None lets the
            standalone model factory use dim. Composite encoder-decoder models derive this
            metadata from the actual encoder. Defaults to ``None``.
        encoder_feature_grid_sizes (list[int | None] | None): Grid side at every encoder
            feature depth. ``None`` creates one entry from ``encoder_output_grid_size``. A
            ``None`` item denotes a flat non-spatial feature. Defaults to ``None``.
        encoder_feature_dims (list[int] | None): Feature width at every encoder depth.
            ``None`` creates one final-feature entry from ``encoder_output_dim``. The list
            index is the ID used by the two encoder aggregation dictionaries. Defaults to
            ``None``.
        shift_inputs (bool): Prepend a learned beginning-of-sequence token and drop the last
            patch token for autoregressive teacher forcing when True. The typed standalone
            dit_decoder factory overrides this field to False for diffusion denoising.
            Defaults to ``True``.
        use_decoder_ids (list[int | None]): Depths implemented with causal- capable
            ``DiTDecoderBlock``. ``[None]`` expands to every decoder depth; ``[]`` selects
            encoder-style blocks. Defaults to a fresh ``[None]`` for each instance.
        decoder_separate_cond (bool): Build independent time/label conditions when True;
            False consumes the encoder condition. The typed standalone dit_decoder factory
            overrides this field to True because no external encoder is present. Defaults to
            ``False``.
        use_causal_mask (bool): Use lower-triangular attention in decoder blocks when True.
            The typed standalone dit_decoder factory overrides this field to False for
            non-autoregressive diffusion denoising. Defaults to ``True``.
        feature_aggregation_ids_dict (dict): Maps a decoder target depth in ``1..depth`` to
            encoder feature IDs. ``-1`` is the final encoder feature and ``None`` selects
            all encoder depths. Example: ``{2: (0, -1)}`` merges the first and final encoder
            features before decoder depth 2. Defaults to a fresh ``{}`` for each instance.
        feature_aggregation_kwargs (dict): Shared ``FeatureHandler`` options:
            ``connect_axis`` (int), ``connect_type`` (``"concat"``/``"add"``),
            ``use_layer_norm`` (bool), ``ln_dim`` (int | None), ``ln_mlp_ratio`` (float |
            None), ``ln_no_adaptation`` (bool), ``mlp_output_dim`` (int | None),
            ``mlp_ratio`` (float | None), and ``mlp_activation_func`` (Keras activation).
            Unknown keys raise ``AssertionError``. Rank-3 token features use axis
            ``1``/``-2`` for tokens and ``2``/``-1`` for channels; flattened rank-2 features
            accept only ``1``/``-1``. Defaults to a fresh ``{}`` for each instance.
        cross_attention_aggregation_ids_dict (dict): Maps decoder depths to encoder features
            used as cross-attention values or queries. It uses the same depth and ID syntax
            as ``feature_aggregation_ids_dict``. Defaults to a fresh ``{}`` for each
            instance.
        cross_attention_aggregation_kwargs (dict): Encoder cross-attention handler options
            with the same accepted keys as ``feature_aggregation_kwargs``. Defaults to a
            fresh ``{}`` for each instance.
    """

    encoder_output_grid_size: int | None = None
    encoder_output_dim: int | None = None
    encoder_feature_grid_sizes: list[int | None] | None = None
    encoder_feature_dims: list[int] | None = None
    shift_inputs: bool = True
    use_decoder_ids: list[int | None] = field(default_factory=lambda: [None])
    decoder_separate_cond: bool = False
    use_causal_mask: bool = True
    feature_aggregation_ids_dict: dict = field(default_factory=dict)
    feature_aggregation_kwargs: dict = field(default_factory=dict)
    cross_attention_aggregation_ids_dict: dict = field(default_factory=dict)
    cross_attention_aggregation_kwargs: dict = field(default_factory=dict)


@dataclass
class DiTEncoderDecoderConfig(DiffusionTransformerConfig):
    """Arguments forwarded to ``DiTEncoderDecoder``.

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory. Inherited fields retain the meanings and
    defaults documented by :class:`DiffusionTransformerConfig`; the attributes below
    document this class's additions and explicit overrides.

    Attributes:
        encoder_kwargs (dict[str, object] | None): Nested encoder constructor values
            (DiffusionTransformer for a generator, DiTClassifier for a classifier). None
            adds no nested overrides. Every inherited typed Config field is also forwarded
            as a flat constructor value and takes precedence over an equal nested key, even
            when that flat value is its dataclass default. The composite forces encoder
            use_unpatchify=False because its decoder owns the image output. Defaults to
            ``None``.
        decoder_kwargs (dict[str, object] | None): :class:`DiTDecoder` arguments. Shared
            class, schedule, image, patch, token-width, and condition-width values default
            to the encoder. Encoder feature dimensions and grids are derived from the actual
            encoder; explicitly supplied metadata must match. A supplied ``build`` value is
            ignored because this model owns symbolic construction. ``shift_inputs`` defaults to False for denoising; True enables right-shifted teacher forcing. The
            decoder also inherits the outer dtype policy unless this mapping provides its
            own ``dtype``. Its image size and channels must match the encoder, and
            ``use_unpatchify`` must be true so the generic diffusion wrapper receives
            image-shaped predictions. ``cond_dim`` must also match unless
            ``decoder_separate_cond=True``. Any decoder timestep/label embedding tables must
            cover the encoder's wrapper-visible ID ranges. Feature-width merges and encoder
            features used as cross-attention queries require matching encoder/decoder class
            and distillation token settings; attention values may differ in length.
            Configure KL bottlenecks and token regularizers on the encoder, because generic
            wrapper losses read the encoder metadata. Defaults to ``None``.
    """

    encoder_kwargs: dict[str, object] | None = None
    decoder_kwargs: dict[str, object] | None = None


@dataclass
class DiTEncoderDecoderClassifierConfig(DiTClassifierConfig):
    """Arguments forwarded to ``DiTEncoderDecoderClassifier``.

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory. Inherited fields retain the meanings and
    defaults documented by :class:`DiTClassifierConfig`; the attributes below
    document this class's additions and explicit overrides.

    Attributes:
        encoder_kwargs (dict[str, object] | None): Nested encoder constructor values
            (DiffusionTransformer for a generator, DiTClassifier for a classifier). None
            adds no nested overrides. Every inherited typed Config field is also forwarded
            as a flat constructor value and takes precedence over an equal nested key, even
            when that flat value is its dataclass default. The composite forces encoder
            use_unpatchify=False because its decoder owns the image output. Defaults to
            ``None``.
        decoder_kwargs (dict[str, object] | None): :class:`DiTDecoder` arguments. Encoder
            feature dimensions and grids are derived from the actual classifier encoder;
            explicitly supplied metadata must match. Shared image, class, timestep, patch,
            token-width, and condition-width settings default to the encoder values. Decoder
            symbolic building is managed by this composite, so a supplied ``build`` value is
            ignored. ``shift_inputs`` defaults to False for denoising; True enables right-shifted teacher forcing. The outer dtype policy is inherited unless ``dtype``
            is set explicitly here. Image size/channels must match the encoder and
            ``use_unpatchify`` must be true. Configure KL bottlenecks and token regularizers
            on the encoder, where unchanged classifier wrappers read their loss metadata.
            ``cond_dim`` must match the encoder unless ``decoder_separate_cond=True``; any
            decoder timestep/label tables must cover the encoder ID ranges. Feature-width
            merges and encoder features used as cross-attention queries require matching
            encoder/decoder class and distillation token settings; attention values may
            differ in length. Defaults to ``None``.
    """

    encoder_kwargs: dict[str, object] | None = None
    decoder_kwargs: dict[str, object] | None = None


@dataclass
class UNetConfig(KwargsMixin):
    """Arguments forwarded to ``UNet``.

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory.

    Attributes:
        num_classes (int | None): Positive number of real classes, or ``None`` for dynamic
            continual growth with CFG. Defaults to ``10``.
        use_cfg (bool): Whether label ID 0 is reserved for CFG. Defaults to ``True``.
        timesteps (int): Positive timestep-embedding vocabulary size. Defaults to ``1000``.
        image_size (int): Positive native square image side. Defaults to ``32``.
        channels (int): Positive input/output channel count. Defaults to ``1``.
        widths (list[int]): Positive encoder widths from high to low spatial resolution.
            Defaults to a fresh ``[32, 64, 96]`` for each instance.
        block_depth (int): Positive residual-stack depth per level. Defaults to ``2``.
        bottleneck_width (int): Positive bottleneck channel width. Defaults to ``128``.
        bottleneck_depth (int): Positive bottleneck residual depth. Defaults to ``2``.
        image_embedding_dim (int): Positive image-projection width. Defaults to ``21``.
        time_embedding_dim (int): Positive timestep-embedding width. Defaults to ``22``.
        label_embedding_dim (int): Positive label-embedding width. Defaults to ``21``.
        activation_func (str): Keras residual activation name. Defaults to ``'swish'``.
        final_activation_func (str): Keras output activation name. Defaults to ``'linear'``.
        use_batch_norm (bool): Enable residual batch normalization. Defaults to ``True``.
        dropout_rate (float): Spatial dropout probability in ``[0,1)``. Defaults to ``0.0``.
        downsampling_method (str): ImageDownsample spatial reduction method: 'max_pooling',
            'avg_pooling', or 'cnn_stride'. Defaults to ``'avg_pooling'``.
        upsampling_method (str): ImageUpsample spatial expansion method: 'interpolate',
            'cnn_interpolate', or 'cnn_transpose'. Defaults to ``'interpolate'``.
        upsampling_interpolation (str): TensorFlow resize method. Defaults to
            ``'bilinear'``.
        use_skip_connections (bool | None): Enable deterministic multiscale feature skips in
            an ordinary U-Net when True. With a variational reshaper, True uses stochastic
            multiscale latent skips; False uses only the central bottleneck. None enables
            skips only when no reshaper is configured. Defaults to ``None``.
        reshaper_ids_dict (dict[int, str]): Optional exact generated flatten/unflatten
            mapping. Defaults to a fresh ``{}`` for each instance.
        reshaper_kwargs (dict): ``add_kl`` and an optional list of positive
            ``latent_dim_ratio`` values, one per generated flatten/unflatten pair in
            ascending depth order. Defaults to a fresh ``{}`` for each instance.
        cls_token_regularizer_ids (list[int | None]): Auxiliary classifier-head depths; an
            empty list disables the heads, while an item None expands across all permitted
            U-Net depths. Defaults to a fresh ``[]`` for each instance.
        cls_token_regularizer_kwargs (dict): Compatibility mapping containing integer
            ``start`` and ``end`` keys. ``train_type`` is ``"normal"``, ``"distil"``, or
            ``"both"``; ``distil_type`` is ``"hard"`` or ``"soft"``. Defaults to a fresh
            ``{'start': 0, 'end': 1, 'train_type': 'normal', 'distil_type': 'hard'}`` for
            each instance.
        extra_depth_specs (list[object]): Serialized progressive stages. Defaults to a fresh
            ``[]`` for each instance.
        name_prefix (str): Prefix for generated Keras layer names. Defaults to ``''``.
        build (bool): Build variables immediately when true. Defaults to ``True``.
    """

    num_classes: int | None = 10
    use_cfg: bool = True
    timesteps: int = 1_000
    image_size: int = 32
    channels: int = 1
    widths: list[int] = field(default_factory=lambda: [32, 64, 96])
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
    """Arguments forwarded to ``UNetClassifier``.

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory. Inherited fields retain the meanings and
    defaults documented by :class:`UNetConfig`; the attributes below
    document this class's additions and explicit overrides.

    Attributes:
        aggregate_from_noises (bool): Classify the predicted noise image instead of saved
            U-Net features. Defaults to ``False``.
        feature_aggregation_ids_dict (dict): Classifier depth-to-U-Net feature routes. Key 1
            supplies the initial classifier feature; later keys inject main features before
            that classifier depth. Main ID 0 is the embedded image, 1..depth are stage
            outputs, and -1 is the final stage. The default route uses only the final main
            feature. Defaults to a fresh ``{1: [-1]}`` for each instance.
        classifier_only_cls_token (bool): Compatibility flag controlling ownership of the
            tracked classifier token placeholder. Defaults to ``False``.
        classifier_only_distil_token (bool): Enable the tracked distillation-token
            placeholder and its parallel softmax head. Defaults to ``False``.
        clf_dim (int | None): Positive classifier width; None uses the last U-Net encoder
            width. Defaults to ``None``.
        clf_depth (int): Nonnegative number of classifier residual stages. Defaults to
            ``1``.
        clf_block_depth (int): Positive residual blocks per classifier stage. Defaults to
            ``1``.
        clf_reshaper_kwargs (dict): Optional classifier ``add_kl`` and a one-entry positive
            ``latent_dim_ratio`` list. Defaults to a fresh ``{}`` for each instance.
        clf_cls_token_regularizer_ids (list[int | None]): Classifier auxiliary-head depth
            IDs. An empty list disables them; an item None expands across all classifier
            depths. Defaults to a fresh ``[]`` for each instance.
        force_global_avg_pooling (bool): Globally pool classifier maps when true; false uses
            resolution-independent global max pooling. Defaults to ``True``.
        classifier_mlp_ratio (float | None): Positive hidden-width ratio, or None to omit
            the hidden Dense layer. Defaults to ``None``.
        classifier_mlp_activation_func (str): Hidden Keras activation name. Defaults to
            ``'tanh'``.
    """

    aggregate_from_noises: bool = False
    feature_aggregation_ids_dict: dict = field(
        default_factory=lambda: {1: [-1]}
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

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory.

    Attributes:
        use_ema (bool): Clone ``network`` from its Keras config and maintain exponential
            moving-average weights after each train step. Deferred raw and cloned networks
            are built before the initial weight copy. Defaults to ``True``.
        test_network_name (str): Default evaluation branch, 'ema' or 'raw'. The model
            factory resolves 'ema' to 'raw' when use_ema=False. Defaults to ``'ema'``.
        ema_decay (float): EMA retention in ``[0,1)``. New EMA weight is ``decay * old_ema +
            (1-decay) * raw``. Defaults to ``0.999``.
        scheduler_name (str): One of ``"linear"``, ``"scaled_linear"``,
            ``"squaredcos_cap_v2"``, ``"clipped_cosine"``, ``"sigmoid"``, ``"quadratic"``,
            ``"ve"``, ``"karras"``, ``"sub_vp"``, or ``"logistic"``. Defaults to
            ``'clipped_cosine'``.
        modify_first_t (bool): Force timestep 0 to have signal rate 1, noise rate 0, and
            cumulative alpha 1 after schedule creation. Defaults to ``False``.
        p_uncond (float): Per-example probability of replacing a shifted class label with
            null ID 0 during training. It is forced to 0 when ``network.use_cfg=False``.
            Defaults to ``0.1``.
        train_cfg_scale (float | None): CFG scale during training. ``None`` runs only the
            conditional network pass; a number additionally runs the null-label pass and
            combines predictions. Defaults to ``None``.
        test_cfg_scale (float): CFG scale for evaluation/sampling, forced to 1 when CFG is
            disabled. Defaults to ``4.0``.
        test_steps (int | None): Default reverse-sampling step count. None lets the model
            factory choose min(50, network.timesteps); an explicit count must satisfy the
            wrapper's sampling bounds and cannot exceed the training horizon. Defaults to
            ``None``.
        test_eta (float): Default stochasticity in ``[0,1]``: 0 is deterministic DDIM and 1
            is DDPM-equivalent only for consecutive full-schedule steps. Defaults to
            ``0.0``.
        noise_loss_coef (float): Multiplier for prediction-versus-noise reconstruction loss
            in ordinary diffusion mode; zero disables this supervised term. In
            swap_noise_image mode the same term reconstructs x_t instead of epsilon.
            Defaults to ``1.0``.
        noise_distil_loss_coef (float): Multiplier for matching a frozen teacher's noise
            prediction on the same noisy inputs; zero disables this teacher term. Positive
            values require an attached teacher or defer_teacher=True until continual
            learning supplies one. Defaults to ``0.0``.
        show_separate_noise_losses (bool): When true, report the unchanged full noise loss
            as ``total_noise_loss`` and additionally report ``cond_noise_loss`` and
            ``uncond_noise_loss`` from non-null and null-label rows. These metrics do not
            change optimization. Defaults to ``False``.
        image_loss_coef (float): Multiplier for reconstructed-image loss; 0 disables it
            during normal training. Defaults to ``0.0``.
        kl_loss_coef (float): Multiplier for variational reshaper KL loss; 0 disables it.
            Defaults to ``0.0``.
        ctr_loss_coef (float): Multiplier for auxiliary class-token regularizer loss; 0
            disables it. Defaults to ``0.0``.
        kl_train_type (str): ``"cond"`` uses conditional latent statistics; ``"uncond"``
            uses the null-label forward pass. Defaults to ``'cond'``.
        ctr_train_type (str): ``"cond"`` or ``"uncond"`` source for auxiliary regularizer
            predictions. ``"uncond"`` requires a non-None ``train_cfg_scale``. Defaults to
            ``'cond'``.
        train_noisified_min_timesteps (int): Inclusive lower timestep bound used when
            preparing ordinary training batches. A zero-width interval at zero yields clean
            images. Defaults to ``0``.
        train_noisified_max_timesteps (int | None): Exclusive training upper bound; -1
            becomes ``network.timesteps`` and None becomes 0. Defaults to ``-1``.
        test_noisified_min_timesteps (int): Inclusive lower timestep bound for ordinary
            evaluation batch preparation. A zero-width interval at zero yields clean images.
            Defaults to ``0``.
        test_noisified_max_timesteps (int | None): Exclusive evaluation upper bound; -1
            becomes ``network.timesteps`` and None becomes 0. Defaults to ``-1``.
        resize_method (str): TensorFlow image-resize method used for progressive-resolution
            preprocessing and reconstruction alignment, such as 'area', 'bilinear', or
            'nearest'. Defaults to ``'area'``.
        resize_antialias (bool): Antialias flag passed to ``tf.image.resize``. Defaults to
            ``True``.
        swap_noise_image (bool): Train the raw output to reconstruct ``x_t`` and route
            the wrapper's sample method to the wrapper's sample_vae method; this mode requires a compatible KL
            bottleneck. Defaults to ``False``.
        map_preprocess (bool): Map ``tf.data.Dataset`` inputs through
            the wrapper's prep_inputs_map method in the wrapper's fit method, the wrapper's evaluate method, and each progressive
            stage. Custom train/test steps then consume the prepared tensors directly. The
            default false preserves online preparation in the training device path. Defaults
            to ``False``.
        map_num_parallel_calls (int | None): Positive parallel-call value forwarded to
            ``Dataset.map``. ``None`` selects ``tf.data.AUTOTUNE``. Defaults to ``1``.
        seen_classes (dict[object, int]): Saved real-label to zero-based classifier-target
            mapping for a grown continual model. ``{}`` starts with no observed classes. A
            nonempty mapping restores dynamic growth and expands a smaller raw/EMA topology
            before checkpoint weights are loaded. The model's dictionary is retained by
            reference in the wrapper config. Defaults to a fresh ``{}`` for each instance.
        seed (int | None): Component TensorFlow seed for noising, label dropout, latent
            draws, and sampling. Per-call seeds can override it. None leaves component
            seeding unset; configured training can inject the experiment seed. Defaults to
            ``None``.
        defer_teacher (bool): Permit a positive teacher objective to start without a teacher
            so continual learning can attach one later. Defaults to ``False``.
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

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory. Inherited fields retain the meanings and
    defaults documented by :class:`DiffusionModelConfig`; the attributes below
    document this class's additions and explicit overrides.

    Attributes:
        clf_distil_type (str): ``"hard"`` applies sparse cross-entropy to the teacher
            argmax; ``"soft"`` applies teacher-to-student KL divergence. Defaults to
            ``'hard'``.
        clf_distil_temperature (float): Positive soft-distillation temperature. ``1``
            preserves the historical direct probability KL exactly; other values soften both
            teacher and student probabilities and apply the standard ``T**2`` scale. Hard
            distillation remains teacher-argmax cross-entropy. Defaults to ``1.0``.
        clf_distil_scope (str): Rows used by teacher-targeted losses: 'old_classes' selects
            labels represented by the frozen teacher; 'replay_only' selects True entries of
            the explicit replay mask in a third dataset tensor; 'current_and_replay' uses
            all current/replay rows, with teacher class support respected when matching
            distributions. Defaults to ``'current_and_replay'``.
        mask_by_nulls (bool | None): Select only examples whose post-dropout CFG label is
            null ID 0 for classifier loss and accuracy when True; False leaves this mask
            disabled. None lets the model factory use network.use_cfg. Enabled masking
            requires p_uncond > 0. Defaults to ``None``.
        mask_by_t_threshold (bool): Additionally select only examples with sampled ``t <=
            filter_t_threshold``. Defaults to ``False``.
        mask_t_percentage (int): Percentage in ``[0,100]`` used to construct the inclusive
            threshold ``ceil(percentage / 100 * timesteps) - 1``. For T=1000 and 70, exactly
            timesteps 0 through 699 are selected; 0 percent selects no examples. Defaults to
            ``70``.
        use_ensemble_loss_instead (bool): Ignore the current forward pass's class
            probabilities for classifier loss and use a four-timestep raw-network ensemble
            on clean images instead. The resulting probabilities are also returned for
            accuracy. Defaults to ``False``.
        clf_train_type (str): ``"cond"`` uses predictions from the conditional/possibly
            dropped-label pass; ``"uncond"`` uses the explicit null-label pass and requires
            ``train_cfg_scale``. Defaults to ``'cond'``.
        clf_loss_coef (float): Scalar multiplier for classifier cross-entropy. Defaults to
            ``0.0086``.
        clf_distil_loss_coef (float): Multiplier for the distillation-token objective; zero
            disables it. Positive values require a distillation head and teacher targets,
            except while defer_teacher allows the initial teacher-free continual task.
            Defaults to ``0.0``.
        clf_acc_coef (float): Primary-head coefficient used only for the wrapper's
            ``total_accuracy`` prediction. Defaults to ``0.5``.
        clf_distil_acc_coef (float): Distillation-head coefficient used only for the
            wrapper's ``total_accuracy`` prediction. Defaults to ``0.5``.
        ctr_acc_coef (float): Classifier-regularizer coefficient used only for the wrapper's
            ``total_accuracy`` prediction. Defaults to ``0.0``.
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
    """Arguments forwarded to ``DiffusionClassifierV2`` except ``network``.

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory. Inherited fields retain the meanings and
    defaults documented by :class:`DiffusionClassifierConfig`; the attributes below
    document this class's additions and explicit overrides.

    Attributes:
        mask_by_nulls (bool): Compatibility override fixed to False by the V2 wrapper. V2
            supplies unconditional labels explicitly in the classifier phase, so null-label
            selection is disabled even if a caller supplies True. Defaults to ``False``.
        clf_loss_coef (float): Multiplier for classifier cross-entropy during the separate
            classifier optimization phase, overriding the smaller joint-training coefficient
            inherited from DiffusionClassifierConfig. Defaults to ``1.0``.
        clf_vars_embedding_ids (list[int]): Shared embeddings assigned to the classifier
            optimizer: 0 patch embedder, 1 time embedder, 2 label embedder, 3 depth-zero
            label regularizer, and 4 a shared main class token. An empty list selects none;
            an item None expands to all five groups. Absent optional layers are skipped.
            Classifier-only tokens are included automatically. Defaults to a fresh ``[]``
            for each instance.
        clf_vars_noise_part_ids (list[int]): Main-network depths assigned to the classifier
            optimizer in addition to its entire classifier branch. Empty selects none. Valid
            IDs are 1..depth or -depth..-1, with -1 selecting the final depth; zero is
            invalid. Negative IDs are resolved again after progressive growth. Defaults to a
            fresh ``[]`` for each instance.
        clf_train_noisified_max_timesteps (int | None): Optional exclusive timestep cap used
            while fitting the classifier part. ``None`` trains on clean images at timestep
            0; ``-1`` uses ``self.timesteps``. Defaults to ``None``.
        clf_test_noisified_max_timesteps (int | None): Optional exclusive timestep cap used
            while evaluating the classifier part. ``None`` evaluates clean images at
            timestep 0; ``-1`` uses ``self.timesteps``. Defaults to ``None``.
    """

    mask_by_nulls: bool = False
    clf_loss_coef: float = 1.0
    clf_vars_embedding_ids: list[int] = field(default_factory=list)
    clf_vars_noise_part_ids: list[int] = field(default_factory=list)
    clf_train_noisified_max_timesteps: int | None = None
    clf_test_noisified_max_timesteps: int | None = None


@dataclass
class VariationalAutoencoderConfig(KwargsMixin):
    """Arguments forwarded to ``VariationalAutoencoder``.

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory.

    Attributes:
        data_dim (int): Positive feature width for input and reconstruction; the expected
            sample shape is ``[batch, data_dim]``. Defaults to ``2048``.
        latent_dim (int): Positive number of Gaussian latent features. Defaults to ``8``.
        hiddens_dims (list[int]): Encoder hidden widths in input-to-latent order; the
            decoder reverses them. An empty list creates direct latent/output projections
            without hidden blocks. Defaults to a fresh ``[16]`` for each instance.
        hiddens_kwargs (dict): Options forwarded to every the raw VAE's dense-layer builder. Allowed keys
            are ``actv`` (a Keras activation name/callable, including the special
            ``"prelu"``), ``use_batch_norm`` (bool), and ``kernel_init`` (a Keras
            initializer name/object). Example: ``{"actv": "relu", "use_batch_norm": False,
            "kernel_init": "glorot_uniform"}``. ``units`` is not valid because each width
            comes from ``hiddens_dims``. Defaults to a fresh ``{}`` for each instance.
        last_activation (str | None): Keras activation for the final reconstruction. Use
            ``"tanh"`` for data scaled to ``[-1, 1]``, ``"sigmoid"`` for ``[0, 1]``, or
            ``None``/ ``"linear"`` for unbounded features. Defaults to ``'tanh'``.
        beta (float): Finite, nonnegative KL-loss multiplier applied to the latent
            regularization term. Defaults to ``0.25``.
        conditioned (bool): Require one-hot labels and concatenate them to encoder/decoder
            inputs when true. Defaults to ``False``.
        class_num (int | None): Positive one-hot width when conditioned. It must be ``None``
            when ``conditioned=False`` and non-``None`` when ``conditioned=True``. Defaults
            to ``None``.
        compile (bool): Whether to call ``Model.compile`` during construction. False leaves
            compilation to the caller. Defaults to ``True``.
        compile_args (dict): Overrides/extends defaults ``{"optimizer":
            Nadam(learning_rate=0.1, decay=0.0), "loss": "mean_squared_error"}``. Keys may
            be any accepted by ``Model.compile``, for example ``{"optimizer": "adam",
            "run_eagerly": True}``. Defaults to a fresh ``{}`` for each instance.
    """

    data_dim: int = 2_048
    latent_dim: int = 8
    hiddens_dims: list[int] = field(default_factory=lambda: [16])
    hiddens_kwargs: dict = field(default_factory=dict)
    last_activation: str | None = "tanh"
    beta: float = 0.25
    conditioned: bool = False
    class_num: int | None = None
    compile: bool = True
    compile_args: dict = field(default_factory=dict)


@dataclass
class VAEClassifierConfig(KwargsMixin):
    """Arguments forwarded to ``VAEClassifier`` except classifier inputs.

    These are YAML-serializable constructor settings; creating this dataclass
    does not build a network. ``kwargs()`` returns a recursively copied mapping
    for the common model factory.

    Attributes:
        data_dim (int): Positive feature width for input and reconstruction; the expected
            sample shape is ``[batch, data_dim]``. Defaults to ``2048``.
        latent_dim (int): Positive number of Gaussian latent features. Defaults to ``8``.
        hiddens_dims (list[int]): Encoder hidden widths in input-to-latent order; the
            decoder reverses them. An empty list creates direct latent/output projections
            without hidden blocks. Defaults to a fresh ``[16]`` for each instance.
        hiddens_kwargs (dict): Options forwarded to every the raw VAE's dense-layer builder. Allowed keys
            are ``actv`` (a Keras activation name/callable, including the special
            ``"prelu"``), ``use_batch_norm`` (bool), and ``kernel_init`` (a Keras
            initializer name/object). Example: ``{"actv": "relu", "use_batch_norm": False,
            "kernel_init": "glorot_uniform"}``. ``units`` is not valid because each width
            comes from ``hiddens_dims``. Defaults to a fresh ``{}`` for each instance.
        last_activation (str | None): Keras activation for the final reconstruction. Use
            ``"tanh"`` for data scaled to ``[-1, 1]``, ``"sigmoid"`` for ``[0, 1]``, or
            ``None``/ ``"linear"`` for unbounded features. Defaults to ``'tanh'``.
        beta (float): Finite, nonnegative KL-loss multiplier applied to the latent
            regularization term. Defaults to ``0.25``.
        alpha (float): Finite, nonnegative coefficient applied to mean categorical
            cross-entropy. Defaults to ``1.0``.
        compile_args (dict): Keras Model.compile overrides for the attached conditional
            VAE/classifier. The raw class merges these with optimizer='adam' and
            loss='mean_squared_error'; common model construction can supply the configured
            optimizer and loss. Defaults to a fresh ``{}`` for each instance.
    """

    data_dim: int = 2_048
    latent_dim: int = 8
    hiddens_dims: list[int] = field(default_factory=lambda: [16])
    hiddens_kwargs: dict = field(default_factory=dict)
    last_activation: str | None = "tanh"
    beta: float = 0.25
    alpha: float = 1.0
    compile_args: dict = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Dataset selection, preprocessing, batching, and optional trial limits.

    Attributes:
        name (str): ``"mnist"``, ``"fmnist"``, ``"cifar10"``, or ``"cifar100"``. Defaults to
            ``'mnist'``.
        preprocess (str | None): ``"min-max"``, ``"normalize"``,
            ``"standardize"``/``"diffusion"``, or no scaling. ``None`` is resolved
            automatically for diffusion and VAE model families. Defaults to ``None``.
        indices (list[int] | None): Original class IDs retained by ordinary dataset
            construction; None retains every class. Continual selection follows
            continually_learn.class_order and task_groups instead. Defaults to ``None``.
        validation_ratio (float): Fraction of training rows reserved for a stratified
            validation split; ``0`` disables the split. Defaults to ``0.0``.
        features_path (str | None): Base path of a saved feature archive without '.npy';
            None provides no explicit archive path. Used only when return_features selects
            the feature-input path. Defaults to ``None``.
        return_features (bool): Use saved features instead of raw images. Defaults to
            ``False``.
        onehot_labels (bool): Return full-width categorical labels instead of sparse IDs.
            Defaults to ``False``.
        max_train_samples (int | None): Positive optional training-row cap; None keeps the
            full prepared split. Continual runs cap once before per-task selection while
            preserving every represented class, so the cap must cover all selected classes.
            Defaults to ``None``.
        max_val_samples (int | None): Positive optional validation-row cap; None keeps the
            full validation split. Caps preserve represented classes and apply only to an
            explicit validation split, never to test rows. Defaults to ``None``.
        batch_size (int): Positive examples per batch. Defaults to ``128``.
        shuffle_buffer (int): Training shuffle-buffer capacity, including each continual
            task. Values above zero enable shuffling; ``0`` disables it. Defaults to
            ``10000``.
        pad (int): Symmetric zero-padding applied to raw images. Continual runs apply it
            before replay; saved features and pretrained/hp-tuned classifiers do not support
            it. Defaults to ``0``.
        trainset_len (int | None): Prepared training-batch count filled by get_datasets for
            optimizer schedule sizing. None means that dataset preparation has not supplied
            the count yet. Defaults to ``None``.
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
        with_classifier (bool): Legacy selector used when name is None: True selects
            DiTClassifier plus DiffusionClassifier; False selects DiffusionTransformer plus
            DiffusionModel. An explicit generic family controls its own compatible wrapper
            path. Defaults to ``True``.
        show_network_summary (bool): Print the wrapper/network summary after construction.
            Defaults to ``True``.
        weights_path (str | None): Keras weights file or TensorFlow checkpoint prefix loaded
            after construction, or ``None`` for fresh weights. In continual runs it
            initializes a VAE replay model, or the classifier and its incremental head
            prefixes for classifier-only and buffer-based runs. A continual diffusion
            checkpoint requires a paired config containing the current ``num_classes`` and
            zero-based wrapper ``seen_classes`` mapping. The continual factory still
            constructs a dynamic raw network and the wrapper grows it before loading these
            weights. Training updates this field to the saved weight artifact path. Defaults
            to ``None``.
        loss_function (str): Generative reconstruction/noise loss passed to Keras
            compilation; ``"mse"`` by default. Defaults to ``'mse'``.
        name (str | None): Generic family selector: 'diffusion_transformer',
            'dit_classifier', 'dit_decoder', 'dit_encoder_decoder',
            'dit_encoder_decoder_classifier', 'unet', 'unet_classifier',
            'vae'/'variational_autoencoder', 'vae_classifier', 'cnn', 'dnn', 'pretrained',
            or 'hp-tuned'. None uses legacy with_classifier to select the compact DiT
            family. Defaults to ``None``.
        wrapper_name (str | None): Explicit diffusion wrapper selector: 'diffusion_model',
            'diffusion_classifier', or 'diffusion_classifier_v2'. None lets the model
            factory infer the compatible process wrapper from the raw family and legacy
            with_classifier setting. Defaults to ``None``.
        kwargs (dict): Generic raw-model or classifier constructor arguments. When nonempty,
            these retain precedence over the typed section for ``name``. Continual diffusion
            construction overrides ``num_classes`` with ``None``. Defaults to a fresh ``{}``
            for each instance.
        wrapper_kwargs (dict): Generic diffusion-wrapper constructor arguments. A nonempty
            mapping replaces the selected typed wrapper section's keyword mapping; omitted
            constructor fields then use wrapper defaults. An empty mapping uses the
            corresponding typed wrapper configuration. Defaults to a fresh ``{}`` for each
            instance.
        classifier_name (str | None): External target-classifier family for continual replay
            and attached VAE classification: 'cnn', 'dnn', 'pretrained', or 'hp-tuned'. None
            defaults to DNN for VAE feature replay/classification and CNN for other replay
            families; a classifier-only model family selects itself. Defaults to ``None``.
        classifier_kwargs (dict): Target-classifier architecture arguments. Defaults to a
            fresh ``{}`` for each instance.
        diffusion_transformer (DiffusionTransformerConfig): Typed DiffusionTransformer
            settings for the matching generic name, or for legacy with_classifier=False.
            Nonempty generic kwargs take precedence. Defaults to a fresh
            ``DiffusionTransformerConfig()`` for each instance.
        dit_classifier (DiTClassifierConfig): Typed DiTClassifier settings for the matching
            generic name, or for legacy with_classifier=True. Nonempty generic kwargs take
            precedence. Defaults to a fresh ``DiTClassifierConfig()`` for each instance.
        dit_decoder (DiTDecoderConfig): Typed DiTDecoder constructor settings selected by
            its matching name. Nonempty generic kwargs take precedence. Defaults to a fresh
            ``DiTDecoderConfig()`` for each instance.
        dit_encoder_decoder (DiTEncoderDecoderConfig): Typed DiTEncoderDecoder constructor
            settings selected by its matching name. Nonempty generic kwargs take precedence.
            Defaults to a fresh ``DiTEncoderDecoderConfig()`` for each instance.
        dit_encoder_decoder_classifier (DiTEncoderDecoderClassifierConfig): Typed
            DiTEncoderDecoderClassifier constructor settings selected by its matching name.
            Nonempty generic kwargs take precedence. Defaults to a fresh
            ``DiTEncoderDecoderClassifierConfig()`` for each instance.
        unet (UNetConfig): Typed UNet constructor settings selected by its matching name.
            Nonempty generic kwargs take precedence. Defaults to a fresh ``UNetConfig()``
            for each instance.
        unet_classifier (UNetClassifierConfig): Typed UNetClassifier constructor settings
            selected by its matching name. Nonempty generic kwargs take precedence. Defaults
            to a fresh ``UNetClassifierConfig()`` for each instance.
        variational_autoencoder (VariationalAutoencoderConfig): Typed VariationalAutoencoder
            settings selected by name='vae' or 'variational_autoencoder'. Nonempty generic
            kwargs take precedence. Defaults to a fresh ``VariationalAutoencoderConfig()``
            for each instance.
        vae_classifier (VAEClassifierConfig): Typed VAEClassifier constructor settings
            selected by its matching name. Nonempty generic kwargs take precedence. Defaults
            to a fresh ``VAEClassifierConfig()`` for each instance.
        diffusion_model (DiffusionModelConfig): Typed DiffusionModel process settings for
            this wrapper selection, including the legacy with_classifier=False path.
            Nonempty wrapper_kwargs take precedence. Defaults to a fresh
            ``DiffusionModelConfig()`` for each instance.
        diffusion_classifier (DiffusionClassifierConfig): Typed joint DiffusionClassifier
            process settings for this wrapper selection, including the legacy
            with_classifier=True path. Nonempty wrapper_kwargs take precedence. Defaults to
            a fresh ``DiffusionClassifierConfig()`` for each instance.
        diffusion_classifier_v2 (DiffusionClassifierV2Config): Typed DiffusionClassifierV2
            constructor settings selected by its matching wrapper_name. Nonempty generic
            wrapper_kwargs take precedence. Defaults to a fresh
            ``DiffusionClassifierV2Config()`` for each instance.
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

        Raises:
            TypeError: If a section is neither a compatible dataclass nor a mapping,
                or a mapping supplies unknown constructor keys.
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
        initial_learning_rate (float): Learning rate at step zero for a cosine schedule, or
            the fixed learning rate for a constant schedule. Defaults to ``0.005``.
        decay_steps (int | None): Positive cosine-decay duration in optimizer steps.
            ``None`` is replaced during model construction with ``epochs * trainset_len``.
            Defaults to ``None``.
        name (str): ``"adam"``, ``"adamw"``, ``"nadam"``, ``"rmsprop"``, or ``"sgd"``.
            Defaults to ``'adam'``.
        schedule (str): 'cosine' uses cosine decay over decay_steps; 'constant' or None
            keeps initial_learning_rate unchanged. Defaults to ``'cosine'``.
        weight_decay (float | None): AdamW-style weight decay; None omits an explicit decay
            setting. Nonzero values require name='adamw' under the supported TensorFlow 2.10
            optimizer API. Defaults to ``None``.
        momentum (float): Momentum used by RMSprop/SGD. Defaults to ``0.0``.
        clipnorm (float | None): Optional positive finite norm used to clip each variable's
            gradient tensor independently. This is Keras ``clipnorm`` semantics, not
            global-gradient clipping. Defaults to ``None``.
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
        class_num (int | None): Number of selected original classes. None infers the count
            from an explicit class_order/task_groups schedule, or uses the dataset's full
            class count when neither is supplied. The resolved count must be at least two
            and within dataset bounds. This sets the task schedule, not a dynamic diffusion
            model's initial head width. Defaults to ``None``.
        class_order (list[int] | None): Unique original dataset labels in introduction
            order. None uses flattened task_groups when supplied, otherwise natural
            zero-based order. An explicit order and explicit groups must agree before
            optional seeded task shuffling. Defaults to ``None``.
        task_groups (list[list[int]] | None): Optional nonempty groups of original labels
            introduced per task. Flattening the groups must equal ``class_order`` when both
            are supplied. ``None`` preserves automatic grouping by ``task_size``. Defaults
            to ``None``.
        task_size (int): Positive number of classes per automatically built task. The
            default ``1`` starts with one class and adds one class at every task. Defaults
            to ``1``.
        class_order_mode (str): ``"fixed"`` or seeded ``"random"`` ordering before automatic
            grouping. Defaults to ``'fixed'``.
        task_order_mode (str): ``"fixed"`` or seeded ``"random"`` ordering of complete
            groups while preserving order inside each group. Defaults to ``'fixed'``.
        seed (int | None): Master continual seed used by schedule resolution, data, model
            initialization, replay, training and sampling. ``None`` falls back to
            ``training.seed`` for backward compatibility. Defaults to ``None``.
        remove_prev_classes (bool): Train later tasks only on their newly introduced classes
            when True; False includes all seen classes. Replay selection may additionally
            supply old examples. Defaults to ``True``.
        keep_same_model (bool): Carry shared classifier weights and old output columns into
            the next expanded classifier. Defaults to ``True``.
        use_loaded_opt (bool): Reconstruct a fresh optimizer from the configuration stored
            with the classifier template. Optimizer slots and its iteration counter are
            intentionally not reused by :func:`common.model.get_model`. Defaults to
            ``True``.
        use_buffer (bool): Use fixed-size sample replay instead of a generative replay
            model. Defaults to ``False``.
        buffer_kwargs (dict[str, object] | None): ReplayBuffer settings maxlen, sample_num,
            insert_num, seed, and strategy ('fifo' or 'reservoir'). None or missing entries
            use the same defaults as the factory mapping. The continual master seed
            overrides the buffer seed when supplied. Defaults to a fresh ``{'maxlen': 10000,
            'sample_num': 1000, 'insert_num': 1000, 'seed': None, 'strategy': 'fifo'}`` for
            each instance.
        baseline (str | None): Optional named treatment: 'sequential', 'cumulative',
            'reservoir_er', 'lwf', 'vae_replay', 'diffusion_replay', 'joint_none',
            'joint_replay', 'joint_kd', or 'joint_both'. Names set coherent
            replay/distillation/classifier switches and validate model compatibility. None
            preserves independently supplied switches. Defaults to ``None``.
        plot_results (bool): Plot accuracy against the number of seen classes. Defaults to
            ``True``.
        generative_model_kwargs (dict[str, int]): Replay-model exposure controls: train_num
            resamples the combined generator training pool to that many rows, while -1 keeps
            the pool; samples_per_class sets generated old examples per class in legacy
            budgeting. Fixed-total mode uses the explicit matched exposure budgets and keeps
            the resulting generator pool. Defaults to a fresh ``{'train_num': 1000,
            'samples_per_class': 1000}`` for each instance.
        use_generative_replay (bool): Generate old examples between tasks when a replay
            model is present. The default ``True`` preserves previous behavior; ``False``
            enables joint/KD controls without generation. Defaults to ``True``.
        replay_budget_mode (str): ``"legacy"`` retains per-class generation and buffer
            counts. ``"fixed_total"`` uses the explicit old/current example budgets below so
            replay methods receive matched exposure. Defaults to ``'legacy'``.
        replay_old_examples (int | None): Exact old-row budget in fixed_total mode, where it
            must be supplied. None is permitted only in legacy mode, which derives counts
            from buffer or per-class generation settings; zero requests no old examples.
            Defaults to ``None``.
        replay_current_examples (int | None): Optional exact current-data exposure per task
            in fixed-total mode; ``None`` keeps every current example. Defaults to ``None``.
        replay_candidate_multiplier (int): Positive candidate-pool multiplier used before an
            optional cognitive replay gate. Defaults to ``1``.
        replay_selection (str): 'all' keeps the legacy candidate selection;
            'uniform'/'random' subsamples uniformly; 'confidence', 'surprise', and
            'confidence_surprise' score candidates using a diffusion-classifier teacher.
            Scored gates require that teacher interface. Defaults to ``'all'``.
        replay_surprise_weight (float): Surprise contribution in the combined
            confidence_surprise gate, between 0 and 1. Other gate modes ignore it. Defaults
            to ``0.5``.
        replay_cache_dir (str | None): Shared directory of authenticated generated-candidate
            archives. None supplies no cache destination and is suitable when
            replay_cache_mode='off'. Defaults to ``None``.
        replay_cache_mode (str): 'off' generates without cache I/O; 'read' requires a
            matching archive; 'write' generates and stores candidates; 'read_write' reuses a
            matching archive or generates and stores one. Defaults to ``'off'``.
        mechanistic_metrics (bool): Compute optional teacher calibration and replay
            consistency/coverage/diversity/drift outcomes per task. Defaults to ``False``.
        mechanistic_max_samples (int): Positive cap on quadratic diversity work. Defaults to
            ``512``.
        use_generative_model_classifier (bool): Use the classifier attached to a diffusion
            replay model instead of the standalone model. Defaults to ``False``.
        train_classifier_separately (bool): Retained for call compatibility; the learner
            selects the separate classifier phase automatically for
            ``DiffusionClassifierV2``. V1 trains both parts jointly. Defaults to ``False``.
        use_distillation (bool): Use each completed diffusion classifier's selected raw/EMA
            snapshot as the next task's frozen teacher. The model must provide a
            distillation token and a positive teacher objective. An optional runtime teacher
            may initialize task one but is never stored in this YAML-safe section. Defaults
            to ``False``.
        snapshot_network_name (str): 'raw' or 'ema' branch cloned for previous-task
            distillation and teacher-scored replay. Selecting 'ema' requires an EMA-enabled
            diffusion wrapper. Defaults to ``'raw'``.
        use_ensemble_accuracy (bool): Use timestep-ensemble values as the authoritative
            continual accuracy matrix and derived metrics. Defaults to ``False``.
        evaluate_ensemble_accuracy (bool): Also record diffusion-classifier ensemble
            accuracy after each continual task. Set ``use_ensemble_accuracy`` to derive
            continual metrics from these scores; that flag also enables their evaluation.
            Defaults to ``False``.
        ensemble_accuracy_kwargs (dict[str, object]): Options for
            DiffusionClassifier.evaluate_ensemble_accuracy/EnsembleAccuracy: max_t is an
            exclusive timestep horizon (default min(128, timesteps)); compute_type is
            'chunked' (default) or 'batched'; t_chunk_size defaults to 16; weighted=False
            gives a uniform mean and True gives normalized SNR weights. network_name selects
            'raw'/'ema'; seed, separate_probas, and classifier/head coefficients can
            override wrapper defaults. The learner supplies a task-specific seed when
            omitted. Defaults to a fresh ``{}`` for each instance.
        return_details (bool): Return task histories and final models together with
            accuracies from a direct configured call. Defaults to ``False``.
        save_task_checkpoints (bool): Persist one atomic recovery checkpoint after every
            completed task. Defaults to ``False``.
        checkpoint_dir (str | None): Optional recovery root. Configured runs default to
            ``<results_path>/checkpoints``. Defaults to ``None``.
        resume_from (str | None): Checkpoint root or committed task directory from which the
            next unfinished task is restored. Defaults to ``None``.
        experiment_phase (str): ``"legacy"`` preserves test reporting, ``"development"``
            prohibits test evaluation, and ``"confirmation"`` enables the frozen
            confirmatory run path. Defaults to ``'legacy'``.
        experiment_manifest_path (str | None): Frozen paired-block manifest required for
            confirmation runs. Defaults to ``None``.
        experiment_manifest_hash (str | None): Trusted external SHA-256 digest used to
            authenticate the confirmation manifest. Defaults to ``None``.
        experiment_run_id (str | None): Planned condition-by-stream run whose schedule and
            seed this invocation must match. Defaults to ``None``.
        optimizer_steps_per_epoch (int | None): Optional positive optimizer update count for
            each active training phase per epoch. ``None`` preserves the existing
            finite-dataset fit behavior. Defaults to ``None``.
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
        default_factory=lambda: {"train_num": 1_000, "samples_per_class": 1_000}
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
        project_tag (str | None): Optional result-run identifier passed to the image
            callback; ``None`` lets that callback choose one. Defaults to ``None``.
        epochs (int): Positive fit epoch count. Defaults to ``20``.
        fit_method (str): Select ``"fit"`` or ``"fit_progressively"``. Progressive training
            uses the curriculum fields below instead of forwarding ``epochs`` to the
            wrapper. Defaults to ``'fit'``.
        stage_tasks (list[object] | str | None): Progressive stage descriptions or
            'timesteps_only', 'resolutions_only', or 'depths_only'. A list's length fixes
            stage count. A task string or two-item operation/value pair changes one of
            timesteps, resolution, or depth; a stage mapping can combine them. YAML lists
            can supply tuple-like values. None is appropriate outside progressive mode;
            progressive mode requires a curriculum. Defaults to ``None``.
        stages_num (int | None): Count used only when shorthand curriculum values must be
            generated. None leaves count inference to the supplied explicit stage list or
            per-stage values; a count is required when neither provides it. Defaults to
            ``None``.
        stages_verbose (bool): Print each resolved progressive stage. Defaults to ``True``.
        stage_epochs (int): Maximum epochs allocated to every listed stage. Defaults to
            ``1``.
        final_epochs (int | None): Epochs for the final full-task stage; ``None`` uses
            ``stage_epochs`` and ``0`` disables that stage. Defaults to ``None``.
        timestep_boundaries (list[list[int] | None] | None): Stage-indexed [inclusive lower,
            exclusive upper] timestep pairs. A stage reads its entry only for a timesteps
            task lacking an inline pair, so unused entries may be None. An omitted list
            generates cumulative easy-to-hard ranges from stages_num. Defaults to ``None``.
        timestep_clustering_type (str): ``"uniform"`` or ``"log_snr"`` clustering used when
            timestep boundaries are generated. Defaults to ``'log_snr'``.
        resolutions (list[int | None] | None): Stage-indexed square image sizes. A stage
            reads its entry only for a resolution task lacking an inline size; unused
            entries may be None. Values may increase, decrease, repeat, or exceed native
            image_size if the network permits them. An omitted list generates low-to-high
            sizes from powers-of-two divisions of image_size. Defaults to ``None``.
        depths (list[object | None] | None): Stage-indexed layer-growth specifications for
            depth tasks without inline values; unused positions may be None. Specifications
            append supported layer dictionaries and completed additions persist after
            fitting. None supplies no explicit per-stage growth list; resolution of
            generated shorthand stages is delegated to fit_progressively. Defaults to
            ``None``.
        pacing_type (str): 'fixed' runs every stage for stage_epochs; 'plateau' may advance
            early when the selected batch-wise or epoch-wise stopping rule sees no
            sufficient improvement. Defaults to ``'fixed'``.
        earlystopping_type (str): Under plateau pacing, 'epoch_wise' uses Keras
            EarlyStopping and 'batch_wise' uses BatchLossPlateau. Fixed pacing does not use
            these stage-advance stoppers. Defaults to ``'epoch_wise'``.
        progressive_monitor (str): Metric forwarded as ``monitor`` to progressive plateau
            stopping. Defaults to ``'val_noise_loss'``.
        progressive_patience (int): Number of non-improving epochs or batches tolerated by
            the selected progressive plateau stopper; the unit follows earlystopping_type.
            Defaults to ``10``.
        min_delta (float): Minimum progressive plateau improvement. Defaults to ``0.001``.
        stopper_mode (str): 'min', 'max', or 'auto' direction passed to Keras EarlyStopping
            for epoch-wise progressive plateau pacing. Defaults to ``'min'``.
        fit_kwargs (dict[str, object]): Additional Keras fit arguments such as step counts;
            each config instance owns an independent mapping. Defaults to a fresh ``{}`` for
            each instance.
        use_valset (bool): Build and pass validation data for the selected dataset when the
            loader created an explicit split. Loaders without a split leave validation
            disabled; test rows are never substituted. Defaults to ``True``.
        show_images (bool): Display callback sample grids during training. Defaults to
            ``False``.
        save_gifs (bool): Save callback denoising animations during training. Trajectory
            callbacks are skipped for no-EMA and VAE/swap models. Defaults to ``True``.
        results_path (str | os.PathLike[str] | None): Base artifact directory passed to the
            image callback. ``None`` is supported only by display-only runs whose runtime
            saving options are all disabled. Defaults to ``'./results'``.
        save_weights (bool): Save final wrapper weights and record their path. Dynamic
            diffusion weights require a paired updated config file; training writes it even
            if ordinary config saving was disabled. Defaults to ``True``.
        task (str): ``legacy``, ``generation``, ``joint``, ``classification``, or
            ``continual``. Defaults to ``'legacy'``.
        seed (int | None): Master TensorFlow/Keras initialization/training and dataset-split
            seed. None leaves ordinary experiment seeding unset; continual runs resolve
            their master seed from continually_learn.seed with this value as fallback.
            Defaults to ``None``.
        dtype_policy (str): Keras global dtype policy installed before data and model
            construction, such as ``"float32"``, ``"mixed_float16"`` or
            ``"mixed_bfloat16"``. Defaults to ``'float32'``.
        deterministic_ops (bool): Request deterministic TensorFlow kernels when supported.
            Continual runs still derive every random source from ``continually_learn.seed``.
            Defaults to ``False``.
        verbose (int): Keras and project reporting verbosity. Defaults to ``1``.
        patience (int): Early-stopping patience; ``0`` disables it. Defaults to ``0``.
        monitor (str | None): Ordinary early-stopping metric. None selects val_loss when an
            explicit validation dataset exists and loss otherwise; progressive pacing uses
            progressive_monitor instead. Defaults to ``None``.
        monitor_mode (str): ``"auto"``, ``"min"``, or ``"max"``. Defaults to ``'auto'``.
        tensorboard (bool): Write TensorBoard summaries when true. Defaults to ``False``.
        tensorboard_path (str | None): TensorBoard root directory. None uses a tensorboard
            subdirectory of the resolved result run. Training appends the project tag (or
            'run') beneath the selected root. Defaults to ``None``.
        tensorboard_run_name (str | None): Event-file suffix. None uses 'run'; HPO supplies
            a compact suffix encoding sampled values or their stable hash. Defaults to
            ``None``.
        report_every_epoch (bool): Enable compatible diffusion image callbacks. Defaults to
            ``True``.
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
        show_history_plot (bool): Display history figures interactively. Defaults to
            ``False``.
        save_history_plot (bool): Save history figures in the result directory. Defaults to
            ``True``.
        final_images_cfg_scale (float): CFG scale for final generation. Defaults to ``3.0``.
        final_images_steps (int): Reverse-diffusion steps for final samples; it must satisfy
            wrapper sampling bounds. Defaults to ``1000``.
        show_final_images (bool): Display the final sample grid. Defaults to ``False``.
        save_final_images (bool): Save the final sample grid as PNG. Defaults to ``True``.
        save_final_gifs (bool): Request sampling trajectories and save a GIF; VAE/swap
            sampling has no trajectories, so only images are reported. Defaults to ``True``.
        plot_without_20percent (bool): Also plot history after discarding the first
            ``int(0.2 * epochs)`` epochs. Defaults to ``True``.
        run_trainset_eval (bool): Evaluate EMA and raw networks on training data. Defaults
            to ``True``.
        run_valset_eval (bool): Evaluate EMA and raw networks on validation data; requires a
            non-``None`` validation dataset. Defaults to ``True``.
        evaluate_ensemble_accuracy (bool): Also evaluate ensemble accuracy for
            ``DiffusionClassifier`` and ``DiffusionClassifierV2`` models. Defaults to
            ``False``.
        ensemble_accuracy_kwargs (dict[str, object]): Options passed to
            DiffusionClassifier.evaluate_ensemble_accuracy, including max_t (default
            min(128, timesteps)), compute_type ('chunked' or 'batched'), t_chunk_size
            (default 16), weighted (default False), separate_probas, seed, and head
            coefficients. Reporting chooses the raw or EMA network for each report,
            overriding a caller network_name. Defaults to a fresh ``{}`` for each instance.
        save_csv (bool): Save epoch metrics and enabled evaluations as CSV. Defaults to
            ``True``.
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

    Section fields accept their declared dataclass or a mapping. Missing fields
    use the independent defaults below; unknown constructor/YAML keys raise
    TypeError. The hpo field is a free-form metadata mapping.

    Attributes:
        dataset (DatasetConfig): Input batching and shuffling settings. Defaults to a fresh
            ``DatasetConfig()`` for each instance.
        model (ModelConfig): Network/wrapper selection and configuration. Defaults to a
            fresh ``ModelConfig()`` for each instance.
        optimizer (OptimizerConfig): Optimizer family, rate/schedule, weight decay,
            momentum, and optional gradient-clipping settings. Defaults to a fresh
            ``OptimizerConfig()`` for each instance.
        training (TrainingConfig): Fit and artifact settings. Defaults to a fresh
            ``TrainingConfig()`` for each instance.
        continually_learn (ContinuallyLearnConfig): Class-incremental loop and replay
            settings. Defaults to a fresh ``ContinuallyLearnConfig()`` for each instance.
        reporting (ReportingConfig): Post-training reporting settings. Defaults to a fresh
            ``ReportingConfig()`` for each instance.
        hpo (dict): Resolved study/trial metadata, sampled values, and selected accuracy
            feedback signal. Defaults to a fresh ``{}`` for each instance.
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

        Raises:
            TypeError: If a section is neither a compatible dataclass nor a mapping,
                or a mapping supplies unknown constructor keys.
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
        path (str | os.PathLike | None): YAML file path. ``None`` returns ``Config()``
            and does not read ``configs/default.yaml``. A YAML file may omit
            sections/fields to receive dataclass defaults, but its root must be a
            mapping. Defaults to ``None``.

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
        are invoked for each comparison so mutable defaults remain isolated. A
        missing default returns (False, None); declared None returns (True, None).
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
        baseline (object): Optional parent-field default instance. ``MISSING`` compares
            fields with their own declarations. Defaults to ``MISSING``.

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
            # Compare nested sections with their parent default when it is a dataclass.
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
        shorten (bool): False saves every declared field. True recursively omits values
            equal to fresh dataclass defaults while retaining changed nested paths. Nonempty
            plain mappings such as hpo remain intact because their entries have no declared
            dataclass defaults. Defaults to ``False``.

    Returns:
        None: The destination contains UTF-8 safe YAML with sorted mapping keys.

    Raises:
        TypeError: If config is not a dataclass instance.
        yaml.representer.RepresenterError: If a field contains a runtime object
            that safe YAML cannot represent, such as a live teacher network.
        OSError: If the destination cannot be opened or written.
    """

    serialized = asdict(config)
    # Prune default fields only for compact YAML output.
    if shorten:
        serialized = _shortened_dataclass(config, serialized)
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(serialized, file, sort_keys=True)
