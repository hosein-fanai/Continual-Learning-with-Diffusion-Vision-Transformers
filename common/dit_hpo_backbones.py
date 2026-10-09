"""Opt-in DiT follow-up templates built from the existing raw-model API.

Local mixers run inside each selected ViT block, after attention and before
the feed-forward sublayer, through the native optional block mixer API.
The plain branch covers the complement of the v10 small-plain capacity domain;
all other branches introduce topology absent from that notebook.
"""

from __future__ import annotations

from typing import Any


FOLLOWUP_BRANCHES = [
    "plain_missing", "u_skip", "local_hybrid", "feature_ladder", 
    "feature_dense", "cross_ladder", "cross_dense", "u_cross"
]
MISSING_CAPACITY_CHOICES = {
    "dim": [256], 
    "depth": [7, 8, 9, 10], 
    "mha_num_heads": [6, 8], 
    "mha_key_dim": ["dim"]
}
CAPACITY_CHOICES = {
    "dim": [16, 32, 64, 128, 256], 
    "depth": [3, 4, 5, 6, 7, 8, 9, 10], 
    "mha_num_heads": [4, 6, 8], 
    "mha_key_dim": [None, "dim"]
}


def suggest_followup_capacity(trial: Any) -> dict[str, object]:
    """Sample a new topology and capacity without dynamic Optuna distributions.

    Plain candidates select an excluded axis under a separate parameter name;
    the other axes retain their complete domains. Thus every plain candidate
    differs from v10 in at least one of width, depth, head count or key width.
    The constrained names accept their own explicit search-space overrides.

    Args:
        trial (object): Optuna-compatible trial, normally the override-aware view.

    Returns:
        dict[str, object]: Branch, missing axis and resolved capacity settings;
            key width retains the symbolic "dim" until raw-model construction.

    Raises:
        ValueError: The trial adapter rejects incompatible categorical overrides.
    """

    branch = trial.suggest_categorical("dit_followup_branch", FOLLOWUP_BRANCHES)
    axis = trial.suggest_categorical(
        "followup_missing_axis", list(MISSING_CAPACITY_CHOICES)
    ) if branch == "plain_missing" else None
    result = {"branch": branch, "missing_axis": axis}
    for parameter, choices in CAPACITY_CHOICES.items():
        # U backbones retain the established symmetric nine-stage graph.
        if parameter == "depth" and branch in ("u_skip", "u_cross"):
            result[parameter] = 9
        # A separately named draw forces the chosen capacity beyond v10.
        elif parameter == axis:
            result[parameter] = trial.suggest_categorical(
                "followup_missing_" + parameter, MISSING_CAPACITY_CHOICES[parameter]
            )
        # Unconstrained capacity axes keep their complete requested domains.
        else:
            result[parameter] = trial.suggest_categorical(parameter, choices)
    return result


def build_followup_backbone(
    branch: str, 
    dim: int, 
    depth: int, 
    local_mixer_variant: str = "separable", 
    local_mixer_kernel_size: int = 3, 
    local_mixer_placement: str = "alternating", 
    feature_merge: str = "concat", 
    cross_merge: str = "concat", 
    cross_plug_type: str = "values", 
    resampling_pos_embed_type: str = "2d_sincos"
) -> dict[str, object]:
    """Resolve a directed, earlier-to-later raw DiT topology.

    Feature routes include the previous stage so routing preserves the current
    stream. Dense routes concatenate or add all earlier transformer outputs.
    Cross routes use decoder blocks, retaining self-attention before attending
    to the selected earlier features. U routes connect matching spatial scales.

    Args:
        branch (str): One of FOLLOWUP_BRANCHES.
        dim (int): Base model width used by connector projections.
        depth (int): At least three processing stages; U templates always use nine.
        local_mixer_variant (str): Depthwise, separable or expanded_pointwise mode.
        local_mixer_kernel_size (int): Spatial kernel side, sampled from 3, 5 or 7.
        local_mixer_placement (str): Every, alternating or late transformer blocks.
        feature_merge (str): Native feature connector add or concat mode.
        cross_merge (str): Native cross-attention source add or concat mode.
        cross_plug_type (str): Plug earlier features into values or queries.
        resampling_pos_embed_type (str): U sampling-stage positional embedding mode.

    Returns:
        dict[str, object]: Raw DiffusionTransformer keyword arguments; no layers
            or tensors are created, and caller inputs are not mutated.

    Raises:
        ValueError: The branch, route depth or local mixer mode is unsupported.
    """

    # Unknown graph names cannot silently fall back to a plain backbone.
    if branch not in FOLLOWUP_BRANCHES:
        raise ValueError(f"Unknown DiT follow-up branch: {branch!r}.")
    # Routed branches need an earlier skip source and a current stream.
    if depth < 3:
        raise ValueError("Follow-up routed backbones require at least three stages.")
    kwargs = {"depth": depth, "dim_forced": True}
    # U variants share the proven two-scale feature-skip topology.
    if branch in ("u_skip", "u_cross"):
        kwargs.update({
            "depth": 9, 
            "dim_forced": False, 
            "connection_ids_dict": {7: [3, 6], 9: [1, 8]}, 
            "vit_block_ids": [1, 3, 5, 7, 9], 
            "vit_block_mlp_output_dims": {
                1: dim, 3: 2 * dim, 5: 2 * dim, 7: dim, 9: dim
            }, 
            "downsample_ids": [2, 4], 
            "downsample_kwargs": {
                "use_layer_norm": True, "ln_no_adaptation": False, 
                "scaling_method": "avg_pooling", 
                "pos_embed_type": resampling_pos_embed_type
            }, 
            "upsample_ids": [6, 8], 
            "upsample_kwargs": {
                "use_layer_norm": True, "ln_no_adaptation": False, 
                "scaling_method": "interpolate", 
                "scaling_interpolation_method": "bilinear", 
                "pos_embed_type": resampling_pos_embed_type
            }
        })
    # Local variants install mixers inside selected attention/FFN blocks.
    if branch == "local_hybrid":
        placements = {
            "every": list(range(1, depth + 1)), 
            "alternating": list(range(1, depth + 1, 2)), 
            "late": list(range(depth // 2 + 1, depth + 1))
        }
        # Only native convolutional modes with an explicit residual projection are supported.
        if local_mixer_variant not in ("depthwise", "separable", "expanded_pointwise"):
            raise ValueError("Unsupported follow-up local-mixer variant.")
        # Reject undefined placement rather than constructing an empty mixer graph.
        if local_mixer_placement not in placements:
            raise ValueError("Unsupported follow-up local-mixer placement.")
        kwargs.update({
            "vit_block_local_mixer_ids": placements[local_mixer_placement], 
            "vit_block_local_mixer_kwargs": {
                "kernel_size": local_mixer_kernel_size, "strides": 1, 
                "depth_multiplier": 1, 
                "use_pointwise": local_mixer_variant != "depthwise", 
                "pointwise_dim_ratio": 2 if local_mixer_variant == "expanded_pointwise" else 1, 
                "use_layer_norm": True, "ln_no_adaptation": False, 
                "zero_init": True, "pos_embed_type": None
            }
        })
    # Feature fusion retains the latest stream alongside earlier-stage outputs.
    if branch in ("feature_ladder", "feature_dense"):
        routes = {
            target: [1, target - 1] if branch == "feature_ladder" else list(range(1, target))
            for target in range(3, depth + 1)
        }
        kwargs.update({
            "connection_ids_dict": routes, 
            "connection_kwargs": {
                "connect_type": feature_merge, "use_layer_norm": True, 
                "mlp_output_dim": dim
            }
        })
    # Decoder blocks preserve self-attention before the routed cross-attention update.
    if branch in ("cross_ladder", "cross_dense", "u_cross"):
        routes = {7: [3], 9: [1]} if branch == "u_cross" else {
            target: [target - 2] if branch == "cross_ladder" else list(range(1, target - 1))
            for target in range(3, depth + 1)
        }
        kwargs.update({
            "cross_attention_ids_dict": routes, 
            "cross_attention_kwargs": {
                "connect_type": cross_merge, "use_layer_norm": True, 
                "mlp_output_dim": dim
            }, 
            "cross_attention_plug_type": cross_plug_type, 
            "use_decoder_ids": list(routes)
        })
    return kwargs


def suggest_followup_topology(
    trial: Any, 
    capacity: dict[str, object], 
    patch_grid: int
) -> dict[str, object]:
    """Suggest active topology options and resolve them to existing model fields.

    Args:
        trial (object): Optuna-compatible trial with optional metadata support.
        capacity (dict[str, object]): Output of suggest_followup_capacity.
        patch_grid (int): Spatial patch-grid side before any resampling stages.

    Returns:
        dict[str, object]: Native raw-model topology options from the sampled branch.

    Raises:
        ValueError: U routing cannot perform two exact spatial reductions, or
            categorical options violate their declared native modes.
    """

    branch = str(capacity["branch"])
    options = {}
    # U variants share the proven two-scale feature-skip topology.
    if branch in ("u_skip", "u_cross"):
        # Two U downsampling levels require integral half and quarter grids.
        if patch_grid % 4:
            raise ValueError("Follow-up U-DiT requires a patch grid divisible by four.")
        options["resampling_pos_embed_type"] = trial.suggest_categorical(
            "resampling_pos_embed_type", ["2d_sincos", "new_weight"]
        )
    # Local variants install mixers inside selected attention/FFN blocks.
    if branch == "local_hybrid":
        options["local_mixer_variant"] = trial.suggest_categorical(
            "local_mixer_variant", ["depthwise", "separable", "expanded_pointwise"]
        )
        options["local_mixer_kernel_size"] = trial.suggest_categorical(
            "local_mixer_kernel_size", [3, 5, 7]
        )
        options["local_mixer_placement"] = trial.suggest_categorical(
            "local_mixer_placement", ["every", "alternating", "late"]
        )
    # Feature fusion retains the latest stream alongside earlier-stage outputs.
    if branch in ("feature_ladder", "feature_dense"):
        options["feature_merge"] = trial.suggest_categorical("feature_merge", ["add", "concat"])
    # Decoder blocks preserve self-attention before the routed cross-attention update.
    if branch in ("cross_ladder", "cross_dense", "u_cross"):
        options["cross_merge"] = trial.suggest_categorical("cross_merge", ["add", "concat"])
        options["cross_plug_type"] = trial.suggest_categorical(
            "cross_plug_type", ["values", "queries"]
        )
    set_user_attr = getattr(trial, "set_user_attr", None)
    # Trial doubles without metadata remain valid suggestion clients.
    if callable(set_user_attr):
        set_user_attr("dit_architecture", branch)
        set_user_attr("dit_followup_missing_axis", capacity["missing_axis"])
    return build_followup_backbone(
        branch, int(capacity["dim"]), int(capacity["depth"]), **options
    )
