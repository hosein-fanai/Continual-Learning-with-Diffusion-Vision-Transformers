"""Configurable diffusion-transformer noise-prediction network.

This module defines the raw TensorFlow network used by the wrappers in
``diffusion.models.wrapper``.  The transformer owns embeddings, token routing,
depth-wise feature processing, and the noise/image output head; wrappers own
the diffusion schedule, noising, losses, optimization, EMA, and sampling.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

import math

from numbers import Real
from typing import Literal, get_args

from . import CondType, TokenType, IdsType, IdsDictType

from common.argument_saver import ArgumentSaverModel
from common.runtime import derive_seed
from common.validation import require

from autoencoder.variational_autoencoder import VariationalAutoencoder

from diffusion.layers.block.vision_transformer_block import VisionTransformerBlock
from diffusion.layers.block.di_t_decoder_block import DiTDecoderBlock
from diffusion.layers.embedding import MergeType
from diffusion.layers.embedding.base_embedding import PosEmbedType
from diffusion.layers.embedding.patch_embedding import PatchEmbedding
from diffusion.layers.embedding.condition_embedding import ConditionEmbedding
from diffusion.layers.manipulation.local_mixer import LocalMixer
from diffusion.layers.manipulation.downsample import Downsample
from diffusion.layers.manipulation.upsample import Upsample
from diffusion.layers.adaptive_layer_normalization_zero import AdaLNZero
from diffusion.layers.feature_handler import FeatureHandler
from diffusion.layers.single_token_layer import SingleTokenLayer


class DiffusionTransformer(ArgumentSaverModel): # DiT
    """Build a configurable, condition-aware diffusion transformer (DiT).

    Images are patchified into a token sequence, optionally merged with time
    and/or label embeddings, processed by ``depth`` ordered stages, and normally
    unpatchified into an image-shaped noise prediction.  Each stage is a
    dictionary of optional components executed in this order: feature
    connector, cross-attention connector, transformer block, local mixer,
    downsampler, upsampler, reshaper, and class-token regularizer.

    Depth numbering is part of the public configuration language.  **Depth
    0** is the embedded input token grid, before any entry in ``layers_dicts``.
    **Depths 1 through N** are the outputs of ``layers_dicts[0]`` through
    ``layers_dicts[N - 1]``.  Thus ``{2: [0, 1]}`` on a connector at depth 2
    combines the input tokens and depth-1 output.  ``None`` in an ID list means
    every eligible depth; a negative ID is normalized as ``id + depth + 1``
    (for total depth 4, ``-1`` is 4 and ``-5`` is 0).  A routed source must
    already exist when its target stage executes.

    ``diffusion.models.transformer`` classes are raw networks.  Pair this class
    with :class:`diffusion.models.wrapper.diffusion_model.DiffusionModel` for
    training, EMA evaluation, forward diffusion, DDIM/DDPM sampling, and
    progressive curricula.

    Attributes:
        layers_dicts (list[dict[str, tf.keras.layers.Layer]]): Per-depth layer
            dictionaries.  Index ``i`` implements architectural depth ``i+1``.
        patch_embedder (PatchEmbedding): Depth-0 image patch embedder.
        time_embedder (ConditionEmbedding | None): Time embedder, created only
            when time is used by conditioning or the class token.
        label_embedder (ConditionEmbedding | None): Label embedder, created only
            when labels are used.  With CFG, label ID 0 is conventionally the
            null class and data labels are shifted to IDs ``1..num_classes`` by
            the wrapper.
        dynamic_num_classes (bool): Whether ``num_classes=None`` requested
            class-by-class vocabulary growth.
        num_classes (int): Current real-class width. After dynamic growth,
            ``get_config()`` records this current value so checkpoint topology
            can be reconstructed at the correct size.
        cls_token (SingleTokenLayer | None): Optional sequence-prefix token.
        distil_token (SingleTokenLayer | None): Optional distillation token,
            placed after ``cls_token`` and before patch tokens.
        unpatchifier (tf.keras.Model): Output projection when
            ``use_unpatchify=True``.
        num_labels (int): ``num_classes + 1`` with CFG, otherwise
            ``num_classes``.
    """

    FC  = "0_feature_connector"
    CAC = "1_cross_attention_connector"
    VTB = "2_vision_transformer_block"
    LM  = "3_local_mixer"
    DS  = "4_downsampler"
    US  = "5_upsampler"
    R   = "6_reshaper"
    CTR = "7_cls_token_regularizer"

    def __init__(
        self, 
        num_classes: int | None = 10, 
        use_cfg: bool = True, 
        timesteps: int = 1_000, 
        image_size: int = 28, 
        channels: int = 1, 
        patch_size: int = 2, 
        dim: int = 32, 
        dim_forced: bool = True, 
        patchify_with_cnn: bool = False, 
        patches_pos_embed_type: PosEmbedType = "2d_sincos", 
        patches_pos_merger_type: MergeType = "add", 
        patches_conds_merger_type: MergeType | None = None, 
        shift_inputs: bool = False, 
        cond_dim: int | None = None, 
        cond_type: CondType | None = "time_label", 
        conds_merger_type: MergeType = "add", 
        time_embed_type: PosEmbedType = "1d_sincos", 
        time_freq_dim: int | None = None, 
        time_embed_trainable: bool = False, 
        time_mlp_ratio: float | None = None, 
        label_embed_type: PosEmbedType = "new_weight", 
        label_embed_trainable: bool = False, 
        label_freq_dim: int | None = None, 
        label_mlp_ratio: float | None = None, 
        cls_token_type: TokenType | None = None, 
        cls_token_freq_dim: int | None = None, 
        cls_token_mlp_ratio: float | None = None, 
        cls_token_pos_merger_type: MergeType = "add", 
        distil_token_type: TokenType | None = None, 
        distil_token_freq_dim: int | None = None, 
        distil_token_mlp_ratio: float | None = None, 
        distil_token_pos_merger_type: MergeType = "add", 
        depth: int = 2, 
        connection_ids_dict: IdsDictType = {}, 
        connection_kwargs: dict = {}, 
        cross_attention_ids_dict: IdsDictType = {}, 
        cross_attention_kwargs: dict = {}, 
        cross_attention_plug_type: Literal["values", "queries"] = "values", 
        vit_block_ids: IdsType = [None], 
        use_decoder_ids: IdsType = [], 
        mha_key_dim: int | None = None, 
        mha_value_dim: int | None = None, 
        mha_num_heads: int = 4, 
        vit_block_mlp_ratio: float = 4., 
        vit_block_mlp_output_dims: dict[int, int] = {},
        ln_mlp_ratio: float | None = None, 
        ln_no_adaptation: bool = False, 
        drop_prob: float = 0., 
        drop_per_sample: bool = True, 
        local_mixer_ids: IdsType = [], 
        local_mixer_kwargs: dict = {}, 
        downsample_ids: IdsType = [], 
        downsample_kwargs: dict = {}, 
        upsample_ids: IdsType = [], 
        upsample_kwargs: dict = {}, 
        reshaper_ids_dict: dict[int, str] = {},
        reshaper_kwargs: dict = {}, 
        cls_token_regularizer_ids: IdsType = [], 
        cls_token_regularizer_kwargs: dict = {
            "start": 0, 
            "end": 1, 
            "train_type": "normal", 
            "distil_type": "hard"
        }, 
        final_ffn_activation_func: str = "linear", 
        use_refiner_cnn: bool = False, 
        refiner_cnn_hidden_dim: int | None = None, 
        refiner_cnn_residual: bool = True, 
        final_activation_func: str = "linear", 
        use_unpatchify: bool = True, 
        name_prefix: str = "", 
        seed: int | None = None, 
        build: bool = True, 
        **kwargs: object
    ) -> None:
        """Initialize the transformer and optionally build all variables.

        Args:
            num_classes (int | None): Positive number of real classes. ``None``
                starts with no real classes and enables class-by-class growth;
                this mode requires classifier-free guidance.
            use_cfg (bool): Whether the label vocabulary includes null label 0
                for classifier-free guidance.  The wrapper shifts real labels
                by one when this is true.
            timesteps (int): Size of the discrete time-embedding vocabulary.
            image_size (int): Base square image size.  It must be divisible by
                ``patch_size``.
            channels (int): Number of image and output channels.
            patch_size (int): Side length of each non-overlapping patch.
            dim (int): Depth-0 patch feature width.
            dim_forced (bool): If true, connectors and spatial layers project
                feature growth back toward the inferred base width when needed.
            patchify_with_cnn (bool): Use a two-convolution ``same``-padding
                stem instead of the single patch-size/stride projection.
            patches_pos_embed_type (PosEmbedType): Patch positional encoding:
                ``"new_weight"``, ``"1d_sincos"``, ``"1d_interpolate"``,
                ``"1d_learned_interpolate"``, ``"2d_sincos"``,
                ``"2d_interpolate"``, or ``"2d_learned_interpolate"``.
            patches_pos_merger_type (MergeType): ``"add"`` preserves width;
                ``"concat"`` appends positional channels.
            patches_conds_merger_type (MergeType | None): How to inject the
                repeated condition into every patch.  ``None`` leaves patches
                separate, ``"add"`` requires ``dim == cond_dim``, and
                ``"concat"`` increases the token width.
            shift_inputs (bool): Prepend the patch embedder's learned BOS token
                and drop the final patch, providing right-shifted decoder input.
            cond_dim (int | None): Combined conditioning width.  ``None`` uses
                ``dim``.  With ``conds_merger_type="concat"`` and both time and
                label conditions, each individual embedder uses half this width.
            cond_type (CondType | None): Adaptive-normalization/patch condition:
                ``"time_label"``, ``"time"``, ``"label"``, or ``None``.
                ``None`` requires ``ln_no_adaptation=True``.
            conds_merger_type (MergeType): Combine simultaneous time and label
                embeddings by ``"add"`` or ``"concat"``.
            time_embed_type (PosEmbedType): Encoding used for integer timesteps.
                Use ``"new_weight"`` or ``"1d_sincos"``; spatial/interpolation
                modes do not produce the rank-2 table required by condition
                lookup.
            time_freq_dim (int | None): Optional pre-MLP time embedding width;
                ``None`` uses the condition-embedder width.
            time_embed_trainable (bool): Whether a non-``"new_weight"`` table
                initialized from the selected time encoding may be trained.
                ``"new_weight"`` is always trainable.
            time_mlp_ratio (float | None): Hidden expansion for the time
                embedding MLP; ``None`` omits its hidden expansion.
            label_embed_type (PosEmbedType): Encoding/table used for label IDs.
                ``"new_weight"`` and ``"1d_sincos"`` are the valid rank-2
                lookup-table modes.
            label_embed_trainable (bool): Whether a non-learned initialized
                label table can train; ``"new_weight"`` is always trainable.
            label_freq_dim (int | None): Optional label embedding width before
                its projection MLP.
            label_mlp_ratio (float | None): Hidden expansion for the label MLP.
            cls_token_type (TokenType | None): ``"new_weight"`` creates a
                learned token; ``"time"``, ``"label"``, or ``"time_label"``
                derives it from those conditions; ``None`` adds no token.
            cls_token_freq_dim (int | None): Optional width of the class-token
                positional representation before projection.
            cls_token_mlp_ratio (float | None): Hidden expansion used by the
                class-token projection.
            cls_token_pos_merger_type (MergeType): ``"add"`` or ``"concat"``
                for the token's learned/positional representation.
            distil_token_type (TokenType | None): Distillation-token source,
                with the same ``"new_weight"``, ``"time"``, ``"label"``,
                ``"time_label"``, and ``None`` choices as ``cls_token_type``.
            distil_token_freq_dim (int | None): Optional width of the
                distillation-token representation before projection.
            distil_token_mlp_ratio (float | None): Hidden expansion used by
                the distillation-token projection.
            distil_token_pos_merger_type (MergeType): Positional merge for the
                distillation token, matching ``cls_token_pos_merger_type``.
            depth (int): Number N of processing stages.  ``0`` creates only
                depth-0 embedding plus the output head; ``1..N`` create those
                many ordered dictionaries in ``layers_dicts``.
            connection_ids_dict (dict[int, list[int | None]]): Maps a target
                depth to feature depths combined before that stage.  Keys are
                in ``1..depth``; source IDs must precede the target after
                normalization.  Example: ``{2: [0, 1]}`` concatenates depth 0
                and depth 1 at stage 2.
            connection_kwargs (dict[str, object]): Options shared by every
                feature connector.  Allowed keys are ``connect_axis`` (int,
                default ``-1``), ``connect_type`` (``"concat"`` or ``"add"``),
                ``use_layer_norm`` (bool), ``ln_dim`` (int | None),
                ``ln_mlp_ratio`` (float | None), ``ln_no_adaptation`` (bool),
                ``mlp_output_dim`` (int | None), ``mlp_ratio`` (float | None),
                and ``mlp_activation_func`` (Keras activation name/callable).
                For ``"add"``, all selected tensors must have identical shape.
            cross_attention_ids_dict (dict[int, list[int | None]]): Same ID
                syntax as ``connection_ids_dict``, but constructs the tensor
                plugged into attention as external queries or values.
            cross_attention_kwargs (dict[str, object]): Same allowed keys and
                value rules as ``connection_kwargs``.
            cross_attention_plug_type (Literal["values", "queries"]): Feed the
                cross-attention connector to the block as keys/values or as
                queries, respectively.
            vit_block_ids (list[int | None]): Stage IDs containing attention
                blocks.  ``[None]`` means every depth, ``[]`` means none, and
                negative IDs are relative to the end.
            use_decoder_ids (list[int | None]): Subset of block IDs implemented
                by ``DiTDecoderBlock``; all other block IDs use encoder blocks.
            mha_key_dim (int | None): Per-head key width; ``None`` lets the
                attention layer infer it.
            mha_value_dim (int | None): Per-head value width; ``None`` uses the
                attention implementation's default.
            mha_num_heads (int): Number of attention heads.
            vit_block_mlp_ratio (float): Transformer FFN hidden expansion.
            vit_block_mlp_output_dims (dict[int, int]): Optional per-depth FFN
                output widths, for example ``{3: 128}``.
            ln_mlp_ratio (float | None): Hidden expansion for adaptive layer
                normalization projections throughout the network.
            ln_no_adaptation (bool): Use ordinary layer normalization without a
                condition-dependent affine/gate path.
            drop_prob (float): Transformer residual-drop probability in
                ``[0, 1]``.
            drop_per_sample (bool): Apply residual dropping independently per
                sample instead of sharing a decision across the batch.
            local_mixer_ids (list[int | None]): Depths with a depthwise spatial
                token mixer.  ID handling matches ``vit_block_ids``.
            local_mixer_kwargs (dict[str, object]): Shared mixer overrides.
                Allowed keys are ``embed_temperature`` (float), ``dim`` (int),
                ``grid_size`` (int), ``use_layer_norm`` (bool),
                ``ln_mlp_ratio`` (float | None), ``ln_no_adaptation`` (bool),
                ``kernel_size`` (int), ``strides`` (int), ``depth_multiplier``
                (int), ``use_pointwise`` (bool), ``pointwise_dim_ratio`` (int),
                ``zero_init`` (bool), ``pos_embed_type`` (a positional type or
                ``None``), ``pos_interpolation_method`` (``tf.image.resize``
                method), ``pos_merger_type`` (``"add"``/``"concat"``), and
                ``mlp_ratio``, ``mlp_activation_func``, ``mlp_output_dim``.
            downsample_ids (list[int | None]): Depths that reduce each spatial
                grid dimension, normally by two.
            downsample_kwargs (dict[str, object]): Allowed keys are
                ``embed_temperature``, ``dim``, ``grid_size``,
                ``use_layer_norm``, ``ln_mlp_ratio``, ``ln_no_adaptation``,
                ``scaling_method`` (``"avg_pooling"``, ``"max_pooling"``, or
                ``"cnn_stride"``), ``cnn_dim_ratio`` (int),
                ``cnn_kernel_size`` (int), ``cnn_activation_func`` (Keras
                activation), positional ``pos_embed_type``,
                ``pos_interpolation_method``, ``pos_merger_type``, and MLP
                ``mlp_ratio``, ``mlp_activation_func``, ``mlp_output_dim``.
            upsample_ids (list[int | None]): Depths that double each spatial
                grid dimension.
            upsample_kwargs (dict[str, object]): Same common embedding, layer
                norm, position, and MLP keys as ``downsample_kwargs`` plus
                ``scaling_method`` (``"cnn_transpose"``, ``"interpolate"``, or
                ``"cnn_interpolate"``), ``scaling_interpolation_method``
                (Keras ``UpSampling2D`` interpolation), ``cnn_dim_ratio``,
                ``cnn_kernel_size``, and ``cnn_activation_func``.
            reshaper_ids_dict (dict[int, str]): Maps depths to ``"flatten"``
                (tokens to one vector) or ``"unflatten"`` (vector back to the
                inferred token grid).  Example: ``{2: "flatten", 4:
                "unflatten"}`` creates a bottleneck between stages 2 and 4.
            reshaper_kwargs (dict[str, object]): Only ``add_kl`` (bool) and
                ``latent_dim_ratio`` (positive float) are allowed.  With
                ``add_kl=True``, a flatten reshaper returns a sampled latent,
                mean, and log variance for a VAE KL objective.
            cls_token_regularizer_ids (list[int | None]): Depth IDs whose token
                slice feeds an auxiliary ``num_classes`` softmax.  ID 0 applies
                a regularizer to the label embedding; ``[None]`` selects 0..N.
            cls_token_regularizer_kwargs (dict[str, object]): ``start`` and
                ``end`` are Python token-slice bounds. Optional ``mlp_ratio``
                adds a hidden Dense layer, and ``activation_function`` selects
                its activation. Missing values default to ``None`` and
                ``"tanh"``, respectively. ``train_type`` is ``"normal"``,
                ``"distil"``, or ``"both"``; ``distil_type`` is ``"hard"``
                or ``"soft"``.
            final_ffn_activation_func (str | callable): Activation on the
                zero-initialized patch-output projection.
            use_refiner_cnn (bool): Add a two-convolution image-space refinement
                head after unpatchification.
            refiner_cnn_hidden_dim (int | None): Refiner hidden channels;
                ``None`` uses the current token width.
            refiner_cnn_residual (bool): Add the refiner output to the initial
                unpatchified image; false returns only the refinement.
            final_activation_func (str | callable): Final output activation.
            use_unpatchify (bool): Return image-shaped output when true; when
                false return final tokens of shape ``[B, tokens, features]``.
            name_prefix (str): Prefix inserted in all generated layer names.
            seed (int | None): Optional raw-network seed used to derive
                independent token-initializer and stochastic-depth streams.
            build (bool): Build symbolic inputs and variables immediately.
            **kwargs (object): Standard ``tf.keras.Model`` options, principally ``name``,
                ``trainable``, ``dtype``, and ``dynamic``.

        Returns:
            None: A configured ``tf.keras.Model`` is initialized in place.
        """

        super().__init__(**kwargs)
        self._check_assertions(locals())
        self._save_init_args(locals())
        self._handle_all_ids()
        self.set_current_resolution()
        derive_seed(self.seed, "diffusion_transformer", "validation")

        self.seed = None if self.seed is None else int(self.seed)
        self.dynamic_num_classes = self.num_classes is None
        self.num_classes = 0 if self.dynamic_num_classes else self.num_classes
        self.num_labels = self.num_classes + int(self.use_cfg)
        self.grid_size = self.image_size // self.patch_size
        self.patches_dim = self.dim
        self.cond_dim = self.dim if self.cond_dim is None else self.cond_dim
        self.dim = self.patches_dim + self.cond_dim if self.patches_conds_merger_type == "concat" else self.dim
        self.cond_embedder_dim = self.cond_dim // 2 if self.conds_merger_type == "concat" and \
                                self.cond_type == "time_label" else self.cond_dim

        self._create_embedders()
        self.cls_token = self._create_single_token(
            self.dim, 
            self.cls_token_pos_merger_type, 
            self.cls_token_freq_dim, 
            self.cls_token_mlp_ratio, 
            self.cls_token_type, 
            name=f"{self.name_prefix}depth_0_cls_token"
        )
        self.distil_token = self._create_single_token(
            self.dim, 
            self.distil_token_pos_merger_type, 
            self.distil_token_freq_dim, 
            self.distil_token_mlp_ratio, 
            self.distil_token_type, 
            name=f"{self.name_prefix}depth_0_distil_token"
        )
        self._create_layers()
        self._create_unpatchifier()

        # Materialize symbolic inputs and variables when eager construction is requested.
        if self.build_:
            self.build()

    def _check_dict_assertions(
        self, 
        local_vars: dict, 
        dict_name: str, 
        check_items_num: bool = True, 
        check_keys: bool = True, 
        allowed_keys: tuple = (), 
        depth_name: str = "depth", 
        second_depth_name: str = "depth", 
        check_values: bool = True, 
        id_less_than_key: bool = True, 
        allowed_values: tuple = (), 
        none_is_filler: bool = True
    ) -> None:
        """Validate a depth-indexed ID mapping or a shared options mapping.

        Args:
            local_vars (dict[str, object]): Namespace containing ``dict_name``
                and the named depth values.
            dict_name (str): Key of the mapping/list to validate.
            check_items_num (bool): Require at most one mapping entry per target
                depth, except when the configured depth is zero.
            check_keys (bool): Validate mapping keys as target depth IDs or
                members of ``allowed_keys``.
            allowed_keys (tuple[object, ...]): Explicit key whitelist.  Empty
                means integer keys in ``1..depth``.
            depth_name (str): Namespace key holding the target-side depth.
            second_depth_name (str): Namespace key holding the source-side
                depth used to validate values.
            check_values (bool): Interpret each mapping value as an ID iterable
                and validate its items.
            id_less_than_key (bool): Require each normalized source ID to be
                less than its target key. This is disabled for cross-branch
                aggregation.
            allowed_values (tuple[object, ...]): Explicit item whitelist.  Empty
                permits the source-depth numeric range.
            none_is_filler (bool): Permit ``None`` as the expand-all sentinel.

        Returns:
            None: Invalid configuration raises ``AssertionError``.

        Note:
            A non-dictionary ID sequence is treated as ``{1: sequence}`` for
            validation.  This helper validates but does not normalize IDs.
        """

        # Limit routed entries to the number of available target depths.
        if check_items_num:
            require(local_vars[depth_name] == 0 or \
                len(local_vars[dict_name]) <= local_vars[depth_name], \
                f"Items (id sets) in {dict_name} cannot be more than {depth_name}.")

        # Normalize a flat ID sequence as a route for depth one.
        if not isinstance((dict_:=local_vars[dict_name]), dict):
            dict_ = {1: dict_}

        for key, value in dict_.items():
            # Validate each route's target depth when requested.
            if check_keys:
                # Use the full target-depth range when no explicit key set was given.
                if len(allowed_keys) == 0:
                    require(local_vars[depth_name] == 0 or \
                        1 <= key <= local_vars[depth_name], \
                        f"Keys in {dict_name} need to be in [1, {local_vars[depth_name]}] range.")
                # Otherwise restrict targets to the caller's allowed key set.
                else:
                    require(key in allowed_keys, \
                        f"Keys in {dict_name} need to be one of {allowed_keys}.")

            # Validate every routed source ID when requested.
            if check_values:
                for id_ in value:
                    # Prevent a connection from reading its own or a future depth.
                    if id_less_than_key:
                        normalized_id = id_ + local_vars[
                            second_depth_name
                        ] + 1 if id_ is not None and id_ < 0 else id_
                        require((none_is_filler and id_ is None) or \
                            normalized_id < key, \
                            f"The ids in each set of {dict_name} can only be less than their key.")

                    # Use the full source-depth range when no explicit set was given.
                    if len(allowed_values) == 0:
                        require(local_vars[second_depth_name] == 0 or \
                            (none_is_filler and id_ is None) or \
                            -(local_vars[second_depth_name]+1) <= id_ <= local_vars[second_depth_name], \
                            f"The ids in each set of {dict_name} can only be None or in "\
                            f"[-{second_depth_name}-1, {second_depth_name}] range.")
                    # Otherwise restrict sources to the caller's allowed value set.
                    else:
                        require(local_vars[second_depth_name] == 0 or \
                        id_ in allowed_values, \
                        f"The ids in each set of {dict_name} can only be one of {allowed_values} .")

    @staticmethod
    def _check_reshaper_kwargs(
        kwargs: dict[str, object], 
        prefix: str = ""
    ) -> None:
        """Validate variational switches shared by both transformer branches.

        Args:
            kwargs (dict[str, object]): Reshaper option mapping after its key
                whitelist has been checked.
            prefix (str): Optional branch name used in assertion messages.

        Returns:
            None: Valid values return normally; invalid values raise
            ``AssertionError``.
        """

        add_kl = kwargs.get("add_kl", False)
        latent_dim_ratio = kwargs.get("latent_dim_ratio", 1.0)

        require(
            isinstance(add_kl, bool), 
            f"{prefix}reshaper add_kl must be boolean."
        )
        require(
            isinstance(latent_dim_ratio, Real) and not isinstance(latent_dim_ratio, bool) and 
            math.isfinite(float(latent_dim_ratio)) and latent_dim_ratio > 0.0, 
            f"{prefix}reshaper latent_dim_ratio must be finite and positive."
        )
    
    def _check_assertions(self, local_vars: dict) -> None:
        """Validate constructor values and record each kwargs whitelist.

        Args:
            local_vars (dict[str, object]): ``__init__`` locals, including all
                ID mappings and layer-option dictionaries.

        Returns:
            None: Invalid dimensions, IDs, option keys, or enum-like values
            raise ``AssertionError``.  The allowed-key tuples are retained on
            the instance for classifier-branch validation.
        """

        require(isinstance(local_vars["image_size"], int) and \
            not isinstance(local_vars["image_size"], bool) and \
            local_vars["image_size"] > 0, \
            "image_size must be a positive integer.")
        require(isinstance(local_vars["patch_size"], int) and \
            not isinstance(local_vars["patch_size"], bool) and \
            local_vars["patch_size"] > 0, \
            "patch_size must be a positive integer.")
        require(local_vars["image_size"] % local_vars["patch_size"] == 0, \
            "image_size must be divisible by patch_size.")
        require(isinstance(local_vars["dim"], int) and \
            not isinstance(local_vars["dim"], bool) and local_vars["dim"] > 0, \
            "dim must be a positive integer.")
        require(isinstance(local_vars["depth"], int) and \
            not isinstance(local_vars["depth"], bool) and local_vars["depth"] >= 0, \
            "depth must be a nonnegative integer.")
        require(isinstance(local_vars["timesteps"], int) and \
            not isinstance(local_vars["timesteps"], bool) and local_vars["timesteps"] >= 2, \
            "timesteps must be an integer greater than or equal to 2.")
        require(local_vars["num_classes"] is None or (
            isinstance(local_vars["num_classes"], int) and
            not isinstance(local_vars["num_classes"], bool) and
            local_vars["num_classes"] > 0
        ), "num_classes must be None or a positive integer.")
        require(local_vars["num_classes"] is not None or local_vars["use_cfg"], \
            "num_classes=None requires use_cfg=True.")
        require(isinstance(local_vars["channels"], int) and \
            not isinstance(local_vars["channels"], bool) and local_vars["channels"] > 0, \
            "channels must be a positive integer.")
        require(isinstance(local_vars["mha_num_heads"], int) and \
            not isinstance(local_vars["mha_num_heads"], bool) and \
            local_vars["mha_num_heads"] > 0, \
            "mha_num_heads must be a positive integer.")
        effective_cond_dim = (
            local_vars["dim"]
            if local_vars["cond_dim"] is None
            else local_vars["cond_dim"]
        )
        require(isinstance(effective_cond_dim, int) and \
            not isinstance(effective_cond_dim, bool) and effective_cond_dim > 0, \
            "cond_dim must be None or a positive integer.")

        # Additive patch conditioning requires equal token and condition widths.
        if local_vars["patches_conds_merger_type"] == "add":
            require(local_vars["dim"] == effective_cond_dim, \
                "When patches_conds_merger_type is add, dim and cond_dim must be equal.")

        # Split concatenated time/label conditions into two equal-width halves.
        if local_vars["conds_merger_type"] == "concat" and \
                local_vars["cond_type"] == "time_label":
            require(effective_cond_dim % 2 == 0, \
                "cond_dim must be even when time and label embeddings are concatenated.")

        # Disable adaptive normalization when no condition tensor exists.
        if local_vars["cond_type"] is None:
            require(local_vars["ln_no_adaptation"], \
                "When cond_type is None, layer_norm cannot use adaptation.")

        require(local_vars["cls_token_type"] in (
            vals:=(None, *get_args(TokenType))), \
            f"cls_token_type can only be one of {vals}.")
        require(local_vars["distil_token_type"] in vals, \
            f"distil_token_type can only be one of {vals}.")

        self._check_dict_assertions(
            local_vars, 
            "connection_ids_dict"
        )
        self._check_dict_assertions(
            local_vars, 
            "connection_kwargs", 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=(feature_handler_kwargs_allowed_vals:=(
                "connect_axis", "connect_type", 
                "use_layer_norm", "ln_dim", 
                "ln_mlp_ratio", "ln_no_adaptation", 
                "mlp_output_dim", "mlp_ratio", 
                "mlp_activation_func", 
            )), 
            check_values=False, 
        ); self.feature_handler_kwargs_allowed_vals = feature_handler_kwargs_allowed_vals
        self._check_dict_assertions(
            local_vars, 
            "cross_attention_ids_dict"
        )
        self._check_dict_assertions(
            local_vars, 
            "cross_attention_kwargs", 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=self.feature_handler_kwargs_allowed_vals, 
            check_values=False, 
        )
        self._check_dict_assertions(
            local_vars, 
            "vit_block_ids", 
            id_less_than_key=False, 
        )
        self._check_dict_assertions(
            local_vars, 
            "use_decoder_ids", 
            id_less_than_key=False, 
        )
        self._check_dict_assertions(
            local_vars, 
            "vit_block_mlp_output_dims", 
            check_values=False
        )
        self._check_dict_assertions(
            local_vars, 
            "local_mixer_ids", 
            id_less_than_key=False, 
        )
        self._check_dict_assertions(
            local_vars, 
            "local_mixer_kwargs", 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=(local_mixer_kwargs_allowed_vals:=(
                "embed_temperature", "dim", "grid_size", 
                "use_layer_norm", "ln_mlp_ratio", 
                "ln_no_adaptation", "kernel_size", 
                "strides", "depth_multiplier", 
                "use_pointwise", "pointwise_dim_ratio", 
                "zero_init", "pos_embed_type", 
                "pos_interpolation_method", 
                "pos_merger_type", "mlp_ratio", 
                "mlp_activation_func", "mlp_output_dim"
            )), 
            check_values=False, 
        ); self.local_mixer_kwargs_allowed_vals = local_mixer_kwargs_allowed_vals
        self._check_dict_assertions(
            local_vars, 
            "downsample_ids", 
            id_less_than_key=False, 
        )
        self._check_dict_assertions(
            local_vars, 
            "downsample_kwargs", 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=(downsample_kwargs_allowed_vals:=(
                "embed_temperature", "dim", "grid_size", 
                "use_layer_norm", "ln_mlp_ratio", 
                "ln_no_adaptation", "scaling_method", 
                "cnn_dim_ratio", "cnn_kernel_size", 
                "cnn_activation_func", "pos_embed_type", 
                "pos_interpolation_method", "pos_merger_type", 
                "mlp_ratio", "mlp_activation_func", 
                "mlp_output_dim"
            )), 
            check_values=False, 
        ); self.downsample_kwargs_allowed_vals = downsample_kwargs_allowed_vals
        self._check_dict_assertions(
            local_vars, 
            "upsample_ids", 
            id_less_than_key=False, 
        )
        self._check_dict_assertions(
            local_vars, 
            "upsample_kwargs", 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=(upsample_kwargs_allowed_vals:=(
                "embed_temperature", "dim", "grid_size", 
                "use_layer_norm", "ln_mlp_ratio", 
                "ln_no_adaptation", "scaling_method", 
                "scaling_interpolation_method", 
                "cnn_dim_ratio", "cnn_kernel_size", 
                "cnn_activation_func", "pos_embed_type", 
                "pos_interpolation_method", "pos_merger_type", 
                "mlp_ratio", "mlp_activation_func", 
                "mlp_output_dim"
            )), 
            check_values=False, 
        ); self.upsample_kwargs_allowed_vals = upsample_kwargs_allowed_vals
        self._check_dict_assertions(
            local_vars, 
            "reshaper_ids_dict", 
            id_less_than_key=False, 
            check_values=False, 
            none_is_filler=False, 
        )
        self._check_dict_assertions(
            local_vars, 
            "reshaper_kwargs", 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=(reshaper_kwargs_allowed_vals:=(
                "add_kl", "latent_dim_ratio"
            )), 
            check_values=False, 
        ); self.reshaper_kwargs_allowed_vals = reshaper_kwargs_allowed_vals
        self._check_reshaper_kwargs(local_vars["reshaper_kwargs"])
        self._check_dict_assertions(
            local_vars, 
            "cls_token_regularizer_ids", 
            id_less_than_key=False, 
            allowed_values=[None]+list(range(local_vars["depth"]+1))
        )
        self._check_dict_assertions(
            local_vars, 
            "cls_token_regularizer_kwargs", 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=(cls_token_regularizer_kwargs_allowed_vals:=(
                "start", "end", "mlp_ratio", "activation_function",
                "train_type", "distil_type"
            )), 
            check_values=False, 
        ); self.cls_token_regularizer_kwargs_allowed_vals = cls_token_regularizer_kwargs_allowed_vals
        regularizer_mlp_ratio = local_vars["cls_token_regularizer_kwargs"].get(
            "mlp_ratio", None
        )
        require(regularizer_mlp_ratio is None or regularizer_mlp_ratio > 0, \
            "regularizer mlp_ratio must be None or positive.")
        require(local_vars["cls_token_regularizer_kwargs"].get(
            "train_type", "normal"
        ) in ("normal", "distil", "both"), \
            "regularizer train_type must be normal, distil, or both.")
        require(local_vars["cls_token_regularizer_kwargs"].get(
            "distil_type", "hard"
        ) in ("hard", "soft"), \
            "regularizer distil_type must be hard or soft.")

        require(local_vars["cross_attention_plug_type"] in ("values", "queries"), \
            "cross_attention_plug_type can only be values or queries.")

    def _fill_none_ids(
        self, 
        ids_dict: dict, 
        min_id: int = 0, 
        max_id: int | None = None
    ) -> dict[int, list[int]]:
        """Expand ID lists containing ``None`` to an inclusive integer range.

        Args:
            ids_dict (dict[int, list[int | None]]): Mapping modified in place.
            min_id (int): First generated ID.
            max_id (int | None): Last generated ID.  ``None`` uses each mapping
                key, so ``{3: [None]}`` becomes ``{3: [0, 1, 2, 3]}`` with the
                default ``min_id``.

        Returns:
            dict[int, list[int]]: The same mapping object after expansion.
        """

        for key in ids_dict:
            max_id_ = key if max_id is None else max_id+1
            ids_dict[key] = list(range(min_id, max_id_)) if None in ids_dict[key] \
                            else ids_dict[key]

        return ids_dict

    def _fix_negative_ids(
        self, 
        ids_dict: dict, 
        depth: int
    ) -> dict[int, list[int]]:
        """Convert negative feature IDs to nonnegative absolute depth IDs.

        Args:
            ids_dict (dict[int, list[int]]): Mapping modified in place.
            depth (int): Maximum depth.  Every negative ID becomes
                ``id + depth + 1``; for depth 4, ``-1`` becomes 4 and ``-5``
                becomes depth 0.

        Returns:
            dict[int, list[int]]: The same normalized mapping.
        """

        for key, value in ids_dict.items():
            value = list(value)

            for i, id_ in enumerate(value):
                # Resolve negative IDs relative to the final model depth.
                if id_ < 0:
                    value[i] = id_ + depth + 1

            ids_dict[key] = value

        return ids_dict

    def _handle_ids(
        self, 
        ids_dict: dict | list, 
        depth: int, min_id: int = 0, 
        max_id: int | None = None
    ) -> dict[int, list[int]] | list[int]:
        """Normalize a mapping or shorthand list of depth IDs.

        Args:
            ids_dict (dict[int, list[int | None]] | list[int | None]): ID
                mapping, or a list shorthand associated with target key 1.
            depth (int): Depth used to resolve negative IDs.
            min_id (int): Lower bound used when expanding ``None``.
            max_id (int | None): Inclusive expansion upper bound.  ``None``
                makes it target-key dependent.

        Returns:
            dict[int, list[int]] | list[int]: The normalized mapping, or a list
            when a list was supplied.  Dictionary inputs are mutated in place.

        Example:
            With ``depth=4``, ``[None]`` and bounds 1..4 becomes
            ``[1, 2, 3, 4]``; ``[-1, -5]`` becomes ``[4, 0]``.
        """

        # Normalize a flat ID sequence as the route for depth one.
        if not_dict:=(not isinstance(ids_dict, dict)):
            ids_dict = {1: ids_dict}

        ids_dict = self._fill_none_ids(
            ids_dict, 
            min_id=min_id, 
            max_id=max_id
        )
        ids_dict = self._fix_negative_ids(
            ids_dict, 
            depth=depth
        )

        # Restore the caller's flat representation after normalization.
        if not_dict:
            return ids_dict[1]
        return ids_dict

    def _handle_all_ids(self) -> None:
        """Normalize every constructor ID collection in place.

        Stage-selection lists expand ``None`` over depths 1..N, class-token
        regularizers expand over 0..N, and routed feature dictionaries expand
        ``None`` from depth 0 through their target key.  Negative IDs use the
        model's total depth.

        Returns:
            None: Corresponding instance attributes are replaced or mutated
            with integer-only IDs.
        """

        self.vit_block_ids = self._handle_ids(
            self.vit_block_ids, 
            depth=self.depth, 
            min_id=1, 
            max_id=self.depth
        )
        self.use_decoder_ids = self._handle_ids(
            self.use_decoder_ids, 
            depth=self.depth, 
            min_id=1, 
            max_id=self.depth
        )
        self.local_mixer_ids = self._handle_ids(
            self.local_mixer_ids, 
            depth=self.depth, 
            min_id=1, 
            max_id=self.depth
        )
        self.downsample_ids = self._handle_ids(
            self.downsample_ids, 
            depth=self.depth, 
            min_id=1, 
            max_id=self.depth
        )
        self.upsample_ids = self._handle_ids(
            self.upsample_ids, 
            depth=self.depth, 
            min_id=1, 
            max_id=self.depth
        )
        self.cls_token_regularizer_ids = self._handle_ids(
            self.cls_token_regularizer_ids, 
            depth=self.depth, 
            min_id=0, 
            max_id=self.depth
        )
        self._handle_ids(
            self.connection_ids_dict, 
            depth=self.depth, 
            max_id=None
        )
        self._handle_ids(
            self.cross_attention_ids_dict, 
            depth=self.depth, 
            max_id=None
        )

    def _get_layers_dict_last_output_dim(
        self, 
        layers_dict: dict, 
        skip_reshaper: bool
    ) -> int | None:
        """Return the final known feature width produced by one stage.

        Args:
            layers_dict (dict[str, tf.keras.layers.Layer]): One ordered stage.
            skip_reshaper (bool): Ignore a reshaper's output width when true.

        Returns:
            int | None: Width from the last dimension-changing component, or
            ``None`` when the stage does not establish a width.
        """

        last_output_dim = None

        # Account for a feature connector's output width.
        if (key:=self.FC) in layers_dict:
            last_output_dim = layers_dict[key].output_dim
        # Account for a transformer block's output width.
        if (key:=self.VTB) in layers_dict:
            last_output_dim = layers_dict[key].output_dim
        # Account for a local mixer's output width.
        if (key:=self.LM) in layers_dict:
            last_output_dim = layers_dict[key].output_dim
        # Account for a downsampler's output width.
        if (key:=self.DS) in layers_dict:
            last_output_dim = layers_dict[key].output_dim
        # Account for an upsampler's output width.
        if (key:=self.US) in layers_dict:
            last_output_dim = layers_dict[key].output_dim
        # Use a reshaper's feature width unless the caller is routing around it.
        if (key:=self.R) in layers_dict and not skip_reshaper:
            last_output_dim = layers_dict[key].output_shape[0][-1]

        return last_output_dim

    def _get_last_output_dim(
        self, 
        i: int, 
        layers_dicts: list[dict], 
        base_dim: int, 
        skip_reshaper: bool = False
    ) -> int:
        """Resolve the most recent feature width at or before stage index ``i``.

        Args:
            i (int): Zero-based stage index; ``-1`` denotes depth 0.
            layers_dicts (list[dict[str, tf.keras.layers.Layer]]): Stages to
                search backward.
            base_dim (int): Width at depth 0.
            skip_reshaper (bool): Ignore reshaper width changes.

        Returns:
            int: Resolved feature width, falling back to ``base_dim``.
        """

        # Return the embedding width before any transformer depth executes.
        if i == -1:
            return base_dim

        last_output_dim = self._get_layers_dict_last_output_dim(
            layers_dicts[i], 
            skip_reshaper
        ) if i < len(layers_dicts) else None
            
        last_output_dim = self._get_last_output_dim(
            i-1, 
            layers_dicts, 
            base_dim, 
            skip_reshaper
        ) if last_output_dim is None else last_output_dim

        return last_output_dim

    def _get_current_output_dim(
        self, 
        i: int, 
        layers_dicts: list[dict], 
        layers_dict: dict, 
        base_dim: int, 
        skip_reshaper: bool = False
    ) -> int:
        """Resolve width after the partially constructed current stage.

        Args:
            i (int): Zero-based current stage index.
            layers_dicts (list[dict]): Completed stages.
            layers_dict (dict): Components already created for this stage.
            base_dim (int): Depth-0 width.
            skip_reshaper (bool): Ignore reshapers while resolving width.

        Returns:
            int: Current inferred feature width.
        """

        layers_dicts = layers_dicts + [layers_dict]
        output_dim = self._get_last_output_dim(
            i, 
            layers_dicts, 
            base_dim, 
            skip_reshaper=skip_reshaper
        )

        return output_dim

    def _get_unforced_total_dim(
        self, 
        ids_set: list[int], 
        layers_dicts: list[dict], 
        base_dim: int, 
        skip_reshaper: bool = False, 
        kwargs: dict | None = None
    ) -> int:
        """Infer the width produced by combining selected feature depths.

        Args:
            ids_set (list[int]): Absolute feature depth IDs; 0 is the input.
            layers_dicts (list[dict]): Source stages.
            base_dim (int): Width of source depth 0.
            skip_reshaper (bool): Ignore reshaper width changes.
            kwargs (dict[str, object] | None): Connector options.  A missing
                ``connect_type`` defaults to ``"concat"``.

        Returns:
            int: Sum of widths for concatenation, the common width for addition,
            or 0 for an empty selection.

        Raises:
            AssertionError: If ``connect_type="add"`` sources have unequal
                widths.
        """

        dims = []

        for i in ids_set:
            # The depth-zero source uses the model's base feature width.
            if i == 0:
                dims.append(base_dim)
            # Later sources use the preceding depth's computed output width.
            else:
                dims.append(self._get_last_output_dim(
                    i-1, 
                    layers_dicts, 
                    base_dim, 
                    skip_reshaper=skip_reshaper
                ))

        # Report zero width when no source features were selected.
        if len(dims) == 0:
            return 0

        # Concatenation combines source widths; additive routes retain one width.
        if kwargs is None or kwargs.get("connect_type", "concat") == "concat":
            return sum(dims)

        for dim_1 in dims:
            for dim_2 in dims:
                require(
                    dim_1 == dim_2, 
                    "In connect_type == add, all of the feature dimensions must be equal."
                )

        return dims[0]

    def _get_last_grid_size(
        self, 
        i: int, 
        layers_dicts: list[dict], 
        base_grid_size: int, 
        skip_reshaper: bool = False
    ) -> int | None:
        """Resolve the latest square token-grid side at or before stage ``i``.

        Args:
            i (int): Zero-based stage index; ``-1`` denotes depth 0.
            layers_dicts (list[dict]): Stages to search backward.
            base_grid_size (int): Depth-0 grid side.
            skip_reshaper (bool): Ignore a reshaper's sequence/vector change.

        Returns:
            int | None: Grid side, or ``None`` when the latest representation
            is a flat vector with no spatial grid.
        """

        # Return the patch grid before any transformer depth executes.
        if i == -1:
            return base_grid_size

        grid_size = None
        # Inspect the requested stage when it exists.
        if i < len(layers_dicts):
            # Read spatial size after a local mixer.
            if (key:=self.LM) in layers_dicts[i]:
                grid_size = layers_dicts[i][key].output_grid_size
            # Read spatial size after downsampling.
            if (key:=self.DS) in layers_dicts[i]:
                grid_size = layers_dicts[i][key].output_grid_size
            # Read spatial size after upsampling.
            if (key:=self.US) in layers_dicts[i]:
                grid_size = layers_dicts[i][key].output_grid_size
            # Infer the grid from a spatial reshaper, or mark flattened output nonspatial.
            if (key:=self.R) in layers_dicts[i] and not skip_reshaper:
                output_shape = layers_dicts[i][key].output_shape[0]
                return int(output_shape[1] ** 0.5) if len(output_shape) == 3 \
                    else None

        grid_size = self._get_last_grid_size(
            i-1, 
            layers_dicts, 
            base_grid_size,
            skip_reshaper=skip_reshaper
        ) if grid_size is None else grid_size

        return grid_size

    def _get_current_grid_size(
        self, 
        i: int, 
        layers_dicts: list[dict], 
        layers_dict: dict, 
        base_grid_size: int, 
        skip_reshaper: bool = False
    ) -> int | None:
        """Resolve grid size after components in the current partial stage.

        Args:
            i (int): Zero-based current stage index.
            layers_dicts (list[dict]): Completed stages.
            layers_dict (dict): Current partially built stage.
            base_grid_size (int): Depth-0 grid side.
            skip_reshaper (bool): Ignore reshapers when resolving the grid.

        Returns:
            int | None: Current spatial grid side.
        """

        layers_dicts = layers_dicts + [layers_dict]
        grid_size = self._get_last_grid_size(
            i, 
            layers_dicts, 
            base_grid_size, 
            skip_reshaper=skip_reshaper
        )

        return grid_size

    def _get_ids_grid_size(
        self, 
        ids_set: list[int], 
        layers_dicts: list[dict], 
        base_grid_size: int, 
        must_be_same: bool = False
    ) -> list[int | None] | int | None:
        """Resolve spatial grid sizes for selected absolute feature IDs.

        Args:
            ids_set (list[int]): Source depths, where 0 is the embedded input.
            layers_dicts (list[dict]): Source stage dictionaries.
            base_grid_size (int): Grid side at depth 0.
            must_be_same (bool): Require equal sides and return one scalar.

        Returns:
            list[int | None] | int | None: One size per source, or their common
            size when ``must_be_same=True``.

        Raises:
            AssertionError: If a common grid was requested but sizes differ.
        """

        grid_sizes = []

        for i in ids_set:
            # The depth-zero source uses the patch grid.
            if i == 0:
                grid_sizes.append(base_grid_size)
            # Later sources inherit the preceding depth's computed grid.
            else:
                grid_sizes.append(self._get_last_grid_size(
                    i=i-1, 
                    layers_dicts=layers_dicts, 
                    base_grid_size=base_grid_size
                ))

        # Return all source grids when equality is not required.
        if not must_be_same:
            return grid_sizes

        for grid_size_1 in grid_sizes:
            for grid_size_2 in grid_sizes:
                require(
                    grid_size_1 == grid_size_2, 
                    "All of the feature grid sizes must be equal."
                )

        return grid_sizes[0]

    def _create_time_embedder(self, name_prefix: str = "") -> ConditionEmbedding:
        """Create the configured discrete timestep embedding layer.

        Args:
            name_prefix (str): Extra component prefix inserted after the model's
                global ``name_prefix``.

        Returns:
            ConditionEmbedding: Layer mapping integer tensors of shape ``[B]``
            to float embeddings of shape ``[B, cond_embedder_dim]``.
        """

        time_embedder = ConditionEmbedding(
            dim=self.cond_embedder_dim, 
            pos_embed_type=self.time_embed_type, 
            embed_steps=self.timesteps, 
            embed_freq_dim=self.time_freq_dim, 
            embed_trainable=self.time_embed_trainable, 
            mlp_ratio=self.time_mlp_ratio, 
            name=f"{self.name_prefix}{name_prefix}depth_0_time_embedder"
        )

        return time_embedder

    def _create_label_embedder(self, name_prefix: str = "") -> ConditionEmbedding:
        """Create the configured class-label embedding layer.

        Args:
            name_prefix (str): Extra component prefix used in the layer name.

        Returns:
            ConditionEmbedding: Layer mapping label IDs of shape ``[B]`` to
            float embeddings of shape ``[B, cond_embedder_dim]``.
        """

        label_embedder = ConditionEmbedding(
            dim=self.cond_embedder_dim, 
            pos_embed_type=self.label_embed_type, 
            embed_steps=self.num_labels, 
            embed_freq_dim=self.label_freq_dim, 
            embed_trainable=self.label_embed_trainable, 
            mlp_ratio=self.label_mlp_ratio, 
            name=f"{self.name_prefix}{name_prefix}depth_0_label_embedder"
        )

        return label_embedder

    def _create_merger(
        self, 
        merger_type: MergeType, 
        name: str | None = None
    ) -> layers.Layer:
        """Create a Keras layer that adds or concatenates tensors.

        Args:
            merger_type (MergeType): ``"add"`` or ``"concat"``.
            name (str | None): Optional Keras layer name.

        Returns:
            tf.keras.layers.Add | tf.keras.layers.Concatenate: Merger layer;
            concatenation always uses the final axis.

        Raises:
            ValueError: If ``merger_type`` is unsupported.
        """

        # Concatenate condition components along their feature axis.
        if merger_type == "concat":
            merger_layer = layers.Concatenate(
                axis=-1, 
                name=name
            )
        # Add condition components elementwise when their widths match.
        elif merger_type == "add":
            merger_layer = layers.Add(
                name=name
            )
        # Reject condition mergers outside the supported alternatives.
        else:
            raise ValueError("conds_merger_type can be either concat or add.")

        return merger_layer

    def _create_embedders(self) -> None:
        """Create all depth-0 patch, condition, and regularizer layers.

        Time and label embedders are instantiated only if requested by
        ``cond_type``, ``cls_token_type``, or ``distil_token_type``.
        ``_cond_type`` becomes empty when no adaptive or patch-level condition
        consumes it. A depth-0 label regularizer is also created when
        regularizer ID 0 is selected.

        Returns:
            None: Embedder and merger attributes are assigned in place.
        """

        self._cond_type = self.cond_type if self.cond_type is not None and \
                        (not self.ln_no_adaptation or self.patches_conds_merger_type is not None) \
                        else []
        self._cls_token_type = self.cls_token_type if self.cls_token_type is not None else []
        self._distil_token_type = self.distil_token_type if self.distil_token_type is not None else []

        embed_times_flag = "time" in self._cls_token_type or \
            "time" in self._distil_token_type or "time" in self._cond_type
        embed_labels_flag = "label" in self._cls_token_type or \
            "label" in self._distil_token_type or \
            "label" in self._cond_type or 0 in self.cls_token_regularizer_ids
        conds_merger_type_flag = \
            ("time" in self._cls_token_type and "label" in self._cls_token_type) or \
            ("time" in self._distil_token_type and "label" in self._distil_token_type) or \
            ("time" in self._cond_type and "label" in self._cond_type)

        self.patch_embedder = PatchEmbedding(
            dim=self.patches_dim, 
            grid_size=self.grid_size, 
            pos_embed_type=self.patches_pos_embed_type, 
            pos_merger_type=self.patches_pos_merger_type, 
            patch_size=self.patch_size, 
            patchify_with_cnn=self.patchify_with_cnn, 
            shift_right_token=self.shift_inputs, 
            seed=derive_seed(self.seed, "patch_embedder"),
            name=f"{self.name_prefix}depth_0_patch_embedder"
        )

        self.time_embedder = self._create_time_embedder(
        ) if embed_times_flag else None

        self.label_embedder = self._create_label_embedder(
        ) if embed_labels_flag else None

        self.conds_merger = self._create_merger(
            merger_type=self.conds_merger_type, 
            name=f"{self.name_prefix}depth_0_time_label_merger"
        ) if conds_merger_type_flag else None

        self.patches_conds_merger = self._create_merger(
            merger_type=self.patches_conds_merger_type, 
            name=f"{self.name_prefix}depth_0_patches_conds_merger_type"
        ) if self.patches_conds_merger_type is not None else None

        self.labels_embed_reg = self._create_token_regularizer(
            i=-1, 
            layers_dicts=[], 
            layers_dict={}, 
            base_dim=self.cond_embedder_dim, 
            kwargs=self.cls_token_regularizer_kwargs, 
            name=f"{self.name_prefix}depth_0_{self.CTR[2:]}"
        ) if 0 in self.cls_token_regularizer_ids else None

    def _create_single_token(
        self, 
        dim: int, 
        pos_merger_type: MergeType, 
        freq_dim: int, 
        mlp_ratio: float, 
        token_type: TokenType, 
        name: str | None = None
    ) -> SingleTokenLayer | None:
        """Create an optional learned or condition-derived prefix token.

        Args:
            dim (int): Output token width.
            pos_merger_type (MergeType): Positional merge operation.
            freq_dim (int): Pre-projection embedding width.
            mlp_ratio (float): Projection hidden expansion.
            token_type (TokenType): ``"new_weight"``, ``"time"``,
                ``"label"``, ``"time_label"``, or ``None``.
            name (str | None): Optional layer name.

        Returns:
            SingleTokenLayer | None: Layer returning one ``[B, 1, dim]`` token,
            or ``None`` when no class token is requested.
        """

        token = SingleTokenLayer(
            dim=dim, 
            pos_merger_type=pos_merger_type, 
            embed_freq_dim=freq_dim, 
            mlp_ratio=mlp_ratio, 
            input_as_token=token_type in (""
                "time_label", "time", "label"
            ), 
            seed=derive_seed(self.seed, "single_token", name or "unnamed"),
            name=name
        ) if token_type is not None else None

        return token

    def _create_feature_handler(
        self, 
        ids_set: list[int], 
        layers_dicts: list[dict], 
        base_dim: int, dim_forced: bool, 
        ln_mlp_ratio: float, ln_no_adaptation: bool, 
        kwargs: dict, zero_index_base_dim: int | None = None, 
        increased_dim: int = 0, output_dim_flag: bool = True, 
        name: str | None = None
    ) -> FeatureHandler:
        """Create a feature selector/merger with an inferred projection width.

        Args:
            ids_set (list[int]): Absolute feature depths selected at call time.
            layers_dicts (list[dict]): Source stages used for width inference.
            base_dim (int): Nominal output width for forced projection.
            dim_forced (bool): Project concatenated growth back to ``base_dim``.
            ln_mlp_ratio (float | None): Adaptive-normalization MLP expansion.
            ln_no_adaptation (bool): Disable condition-dependent normalization.
            kwargs (dict[str, object]): Feature-handler overrides.  Accepted keys
                are documented on ``connection_kwargs``; supplied values take
                precedence over inferred values.
            zero_index_base_dim (int | None): Alternate width for source ID 0.
            increased_dim (int): Width of an additional ``second_list`` tensor.
            output_dim_flag (bool): Permit automatic forced projection.
            name (str | None): Keras layer name.

        Returns:
            FeatureHandler: A layer accepting a feature list, optional second
            list and condition, and returning merged rank-3 or rank-2 features.

        Raises:
            AssertionError: If additive sources have incompatible widths.
        """

        increased_dim_ = self._get_unforced_total_dim(
            ids_set, 
            layers_dicts, 
            base_dim=base_dim if zero_index_base_dim is None \
                    else zero_index_base_dim, 
            kwargs=kwargs
        )
        # Sum connector width growth when features are concatenated.
        if kwargs.get("connect_type", "concat") == "concat":
            increased_dim_ += increased_dim
        # Additive features must all contribute the same width.
        elif increased_dim != 0:
            require(
                increased_dim_ == increased_dim, 
                "In connect_type == add, all of the feature dimensions must be equal."
            )

        mlp_output_dim = base_dim if dim_forced and \
                        increased_dim_ > base_dim and \
                        output_dim_flag else None

        feature_handler_kwargs = {
            "ids": ids_set, 
            "ln_dim": increased_dim_, 
            "mlp_output_dim": mlp_output_dim, 
            "ln_mlp_ratio": ln_mlp_ratio, 
            "ln_no_adaptation": ln_no_adaptation, 
            "name": name
        }
        feature_handler_kwargs.update(kwargs)

        feature_handler = FeatureHandler(
            **feature_handler_kwargs
        )

        return feature_handler

    def _create_vit_block(
        self, 
        i: int, 
        layers_dicts: list[dict], 
        layers_dict: dict, 
        base_dim: int, 
        mha_key_dim: int | None, 
        mha_value_dim: int | None, 
        mha_num_heads: int, 
        mlp_ratio: float, 
        mlp_output_dim: int, 
        ln_mlp_ratio: float, 
        ln_no_adaptation: bool, 
        drop_prob: float, 
        drop_per_sample: bool, 
        use_decoder: bool, 
        name_prefix: str, 
        mha_query_dim: int | None = None
    ) -> VisionTransformerBlock | DiTDecoderBlock:
        """Create one encoder- or decoder-style transformer block.

        Args:
            i (int): Zero-based current stage index.
            layers_dicts (list[dict]): Completed stages.
            layers_dict (dict): Current partially built stage.
            base_dim (int): Depth-0 feature width.
            mha_key_dim (int | None): Per-head key width.
            mha_value_dim (int | None): Per-head value width.
            mha_num_heads (int): Attention-head count.
            mlp_ratio (float): Feed-forward hidden expansion.
            mlp_output_dim (int | None): Optional changed block output width.
            ln_mlp_ratio (float | None): Adaptive-normalization MLP expansion.
            ln_no_adaptation (bool): Disable condition adaptation.
            drop_prob (float): Residual-drop probability.
            drop_per_sample (bool): Whether dropping is sample-wise.
            use_decoder (bool): Create ``DiTDecoderBlock`` when true, otherwise
                ``VisionTransformerBlock``.
            name_prefix (str): Full generated Keras layer-name prefix.
            mha_query_dim (int | None): External query width when known.

        Returns:
            VisionTransformerBlock | DiTDecoderBlock: A block mapping token and
            condition tensors to a token tensor.
        """

        block_kwargs = {
            "dim": self._get_current_output_dim(
                i, 
                layers_dicts, 
                layers_dict, 
                base_dim
            ), 
            "key_dim": mha_key_dim, 
            "value_dim": mha_value_dim, 
            "query_dim": mha_query_dim, 
            "num_heads": mha_num_heads, 
            "mlp_ratio": mlp_ratio, 
            "mlp_output_dim": mlp_output_dim, 
            "ln_mlp_ratio": ln_mlp_ratio, 
            "ln_no_adaptation": ln_no_adaptation, 
            "drop_prob": drop_prob, 
            "drop_per_sample": drop_per_sample, 
            "seed": derive_seed(
                self.seed,
                "transformer_block",
                i,
                name_prefix,
            ),
            "name": name_prefix
        }

        # Build a conditioned decoder block when cross-attention is requested.
        if use_decoder:
            block_kwargs["name"] += "decoder_block"
            block = DiTDecoderBlock(**block_kwargs)
        # Otherwise build a standard vision-transformer block.
        else:
            block_kwargs["name"] += "encoder_block"
            block = VisionTransformerBlock(**block_kwargs)
            
        return block

    def _create_local_mixer(
        self, 
        i: int, 
        dim_forced: bool, 
        layers_dicts: list[dict], 
        layers_dict: dict, 
        base_dim: int, 
        base_grid_size: int, 
        ln_mlp_ratio: float, 
        ln_no_adaptation: bool, 
        circumvent_tokens: bool | int, 
        kwargs: dict = {}, 
        name: str | None = None
    ) -> LocalMixer:
        """Create a depthwise-convolution local token mixer.

        The mixer reshapes square patch tokens to an image grid, performs a
        depthwise convolution and optional pointwise projection, restores the
        sequence, and uses a residual only when ``strides == 1``.  A class token
        is temporarily removed when ``circumvent_tokens=True``.

        Args:
            i (int): Zero-based current stage index.
            dim_forced (bool): Force positional/channel growth back to input
                width when required.
            layers_dicts (list[dict]): Completed stages.
            layers_dict (dict): Current partial stage.
            base_dim (int): Depth-0 feature width.
            base_grid_size (int): Depth-0 grid side.
            ln_mlp_ratio (float | None): Normalization MLP expansion.
            ln_no_adaptation (bool): Disable condition adaptation.
            circumvent_tokens (bool | int): Number of leading special tokens
                kept outside spatial convolution.
            kwargs (dict[str, object]): Allowed ``local_mixer_kwargs`` options.
            name (str | None): Keras layer name.

        Returns:
            LocalMixer: Configured rank-3 token mixer.
        """

        local_mixer_kwargs = {
            "dim": self._get_current_output_dim(
                i, 
                layers_dicts, 
                layers_dict, 
                base_dim
            ), 
            "grid_size": self._get_current_grid_size(
                i, 
                layers_dicts, 
                layers_dict, 
                base_grid_size=base_grid_size
            ), 
            "ln_mlp_ratio": ln_mlp_ratio, 
            "ln_no_adaptation": ln_no_adaptation, 
            "circumvent_tokens": circumvent_tokens, 
            "name": name
        }
        local_mixer_kwargs.update(kwargs)

        flag1 = kwargs.get("pos_merger_type", "add") == "concat" and \
                kwargs.get("pos_embed_type", "new_weight") is not None
        flag2 = kwargs.get("depth_multiplier", 1) > 1 and not kwargs.get("use_pointwise", True)
        # Project mixer output back to the forced width after width-changing options.
        if dim_forced and (flag1 or flag2):
            local_mixer_kwargs["mlp_output_dim"] = local_mixer_kwargs["dim"]

        cnn = LocalMixer(**local_mixer_kwargs)

        return cnn

    def _create_scaler(
        self, 
        scaler_type: str, 
        i: int, 
        dim_forced: bool, 
        layers_dicts: list[dict], 
        layers_dict: dict, 
        base_dim: int, 
        base_grid_size: int, 
        ln_mlp_ratio: float, 
        ln_no_adaptation: bool, 
        circumvent_tokens: bool | int, 
        kwargs: dict = {}, 
        name: str | None = None
    ) -> Downsample | Upsample:
        """Create a token-grid downsampler or upsampler.

        Args:
            scaler_type (str): Exactly ``"downsample"`` or ``"upsample"``.
            i (int): Zero-based current stage index.
            dim_forced (bool): Project channel/position growth back to inferred
                input width when needed.
            layers_dicts (list[dict]): Completed stages.
            layers_dict (dict): Current partial stage.
            base_dim (int): Depth-0 feature width.
            base_grid_size (int): Depth-0 spatial grid side.
            ln_mlp_ratio (float | None): Normalization MLP expansion.
            ln_no_adaptation (bool): Disable condition adaptation.
            circumvent_tokens (bool | int): Number of leading special tokens
                kept outside resizing.
            kwargs (dict[str, object]): The corresponding documented
                downsample/upsample option mapping.
            name (str | None): Keras layer name.

        Returns:
            Downsample | Upsample: Layer mapping ``[B, G*G(+1), D]`` tokens to
            a resized token grid.

        Raises:
            ValueError: If ``scaler_type`` is unsupported.
        """

        scaler_kwargs = {
            "dim": self._get_current_output_dim(
                i, 
                layers_dicts, 
                layers_dict, 
                base_dim
            ), 
            "grid_size": self._get_current_grid_size(
                i, 
                layers_dicts, 
                layers_dict, 
                base_grid_size=base_grid_size
            ), 
            "ln_mlp_ratio": ln_mlp_ratio, 
            "ln_no_adaptation": ln_no_adaptation, 
            "circumvent_tokens": circumvent_tokens, 
            "name": name
        }
        scaler_kwargs.update(kwargs)

        flag1 = kwargs.get("pos_merger_type", "add") == "concat" and \
                kwargs.get("pos_embed_type", "new_weight") is not None
        default_method = "avg_pooling" if scaler_type == "downsample" \
                        else "cnn_transpose"
        scaling_method = kwargs.get("scaling_method", default_method)
        width_changing_methods = ("cnn_stride",) if scaler_type == "downsample" \
                                else ("cnn_transpose", "cnn_interpolate")
        flag2 = kwargs.get("cnn_dim_ratio", 1) > 1 and \
                scaling_method in width_changing_methods
        # Project scaler output back to the forced width after width-changing options.
        if dim_forced and (flag1 or flag2):
            scaler_kwargs["mlp_output_dim"] = scaler_kwargs["dim"]

        # Construct the requested downsampling layer.
        if scaler_type == "downsample":
            scaler = Downsample(**scaler_kwargs)
        # Construct the requested upsampling layer.
        elif scaler_type == "upsample":
            scaler = Upsample(**scaler_kwargs)
        # Reject scaler directions outside the supported pair.
        else:
            raise ValueError("scaler_type can either be downsample or upsample.")

        return scaler

    def _resize_reshaper_tokens(
        self, 
        x: tf.Tensor, 
        input_grid_size: int | None, 
        output_grid_size: int, 
        dim: int, 
        grid_has_tokens: bool | int
    ) -> tf.Tensor:
        """Resize token grids around a base-resolution reshape operation.

        Args:
            x (tf.Tensor): Float tensor ``[B, tokens, dim]``.
            input_grid_size (int | None): Source grid side; ``None`` infers it
                from the token count after removing any class token.
            output_grid_size (int): Target grid side.
            dim (int): Static feature width used for reshape/set-shape.
            grid_has_tokens (bool | int): Number of leading non-spatial
                tokens to preserve; booleans retain the one-token API.

        Returns:
            tf.Tensor: Resized tokens of shape
            ``[B, output_grid_size**2 + int(grid_has_tokens), dim]``. At the
            base image resolution the input tensor is returned unchanged.
        """

        # Avoid an identity resize at the network's native resolution.
        if self._current_resolution == self.image_size:
            return x

        x, token = (
            x[:, int(grid_has_tokens):, :], 
            x[:, :int(grid_has_tokens), :]
        ) if grid_has_tokens else (x, None)

        x_shape = tf.shape(x)
        input_grid_size = tf.cast(
            tf.sqrt(tf.cast(
                x_shape[1], dtype=self.dtype_policy.variable_dtype
            )),
            dtype=tf.int32
        ) if input_grid_size is None else input_grid_size

        x = tf.reshape(x, (
            x_shape[0], 
            input_grid_size, 
            input_grid_size, 
            dim
        ))
        x = tf.image.resize(x,
            size=(
                output_grid_size, 
                output_grid_size
            )
        )
        x = tf.reshape(x, (
            x_shape[0], 
            output_grid_size * output_grid_size, 
            dim
        ))
        x.set_shape(
            (None, output_grid_size * output_grid_size, dim)
        )

        x = tf.concat([
            token, x
        ], axis=1) if grid_has_tokens else x
        x.set_shape((
            None, 
            output_grid_size * output_grid_size + int(grid_has_tokens), 
            dim
        ))

        return x

    def _create_reshaper(
        self, 
        reshape_type: str, 
        i: int, 
        layers_dicts: list[dict], 
        layers_dict: dict, 
        base_dim: int, 
        base_grid_size: int, 
        grid_has_tokens: bool | int, 
        kwargs: dict = {}, 
        name: str | None = None
    ) -> models.Model:
        """Create a token/vector bottleneck model for one stage.

        Args:
            reshape_type (str): ``"flatten"`` maps ``[B, tokens, dim]`` to
                ``[B, tokens*dim]``; ``"unflatten"`` performs the inverse.
            i (int): Zero-based current stage index.
            layers_dicts (list[dict]): Completed stages.
            layers_dict (dict): Current partial stage.
            base_dim (int): Depth-0 feature width.
            base_grid_size (int): Depth-0 grid side.
            grid_has_tokens (bool | int): Number of prefix tokens included
                in reshape sizes; booleans retain the one-token API.
            kwargs (dict[str, object]): ``add_kl`` (bool) and
                ``latent_dim_ratio`` (float, default 1).
            name (str | None): Required/generated model name.

        Returns:
            tf.keras.Model: Model returning ``(x, mean, log_variance)``.  Without
            a KL flatten bottleneck, the latter two outputs are dummy scalar
            batch-size tensors.  With ``add_kl=True`` on ``"flatten"``, ``x``
            is a reparameterized latent projected back to the flattened width.

        Raises:
            ValueError: If ``reshape_type`` is not ``"flatten"`` or
                ``"unflatten"``.
        """

        grid_size = self._get_current_grid_size(
            i=i, 
            layers_dicts=layers_dicts, 
            layers_dict=layers_dict, 
            base_grid_size=base_grid_size, 
            skip_reshaper=True
        )
        dim = self._get_current_output_dim(
            i=i, 
            layers_dicts=layers_dicts, 
            layers_dict=layers_dict, 
            base_dim=base_dim, 
            skip_reshaper=True
        )
        shape1 = (
            (grid_size * grid_size + int(grid_has_tokens)) * dim, 
        )
        shape2 = (
            grid_size * grid_size + int(grid_has_tokens), 
            dim
        )

        # Configure the spatial-to-token flattening direction.
        if reshape_type == "flatten":
            source_shape = shape2
            target_shape = shape1
        # Configure the token-to-spatial unflattening direction.
        elif reshape_type == "unflatten":
            source_shape = shape1
            target_shape = shape2
        # Reject reshaping directions outside flatten and unflatten.
        else:
            raise ValueError(
                "reshape_type needs to be either flatten or unflatten."
            )

        reshaper_layer = layers.Reshape(
            target_shape, 
            dtype=self.dtype_policy,
            name=name
        )

        inputs = layers.Input(
            shape=(None, dim) if reshape_type == "flatten" else source_shape,
            dtype=self.compute_dtype,
        )


        def resize_to_base(x: tf.Tensor) -> tf.Tensor:
            """Resize an active-resolution token grid to the base grid.

            Args:
                x (tf.Tensor): Float token tensor ``[B, tokens, dim]``.

            Returns:
                tf.Tensor: Float tokens on the constructor-time base grid.
            """

            return self._resize_reshaper_tokens(
                x, 
                input_grid_size=None, 
                output_grid_size=grid_size, 
                dim=dim, 
                grid_has_tokens=grid_has_tokens
            )


        def resize_from_base(x: tf.Tensor) -> tf.Tensor:
            """Resize a base-grid token sequence to the active grid.

            Args:
                x (tf.Tensor): Float token tensor ``[B, tokens, dim]``.

            Returns:
                tf.Tensor: Float tokens on the active-resolution grid.
            """

            return self._resize_reshaper_tokens(
                x, 
                input_grid_size=grid_size, 
                output_grid_size=(
                    grid_size * self._current_resolution // self.image_size
                ), 
                dim=dim, 
                grid_has_tokens=grid_has_tokens
            )


        x = layers.Lambda(
            resize_to_base,
            dtype=self.dtype_policy,
            name=name+"/resize_to_base"
        )(inputs) if reshape_type == "flatten" else inputs
        x = reshaper_layer(x)
        x = layers.Lambda(
            resize_from_base,
            dtype=self.dtype_policy,
            name=name+"/resize_from_base"
        )(x) if reshape_type == "unflatten" else x

        # Add a variational latent projection only on KL-enabled flatten stages.
        if kwargs.get("add_kl", False) and reshape_type == "flatten":
            latent_dim_ratio = kwargs.get("latent_dim_ratio", 1)
            latent_dim = int(target_shape[-1] * latent_dim_ratio)

            # A positive fractional ratio can still truncate to an unusable
            # zero-width latent for a small flattened feature.
            if latent_dim < 1:
                raise ValueError(
                    "latent_dim_ratio creates an empty latent vector."
                )

            z_mean = layers.Dense(
                latent_dim, 
                dtype=self.dtype_policy,
                name=name+"/z_mean"
            )(x)
            z_log_var = layers.Dense(
                latent_dim, 
                dtype=self.dtype_policy,
                name=name+"/z_log_var"
            )(x)
            z = VariationalAutoencoder.compute_z(
                z_mean,
                z_log_var,
                seed=derive_seed(
                    self.seed,
                    "reshaper",
                    name,
                    "reparameterization",
                ),
                dtype=self.dtype_policy.variable_dtype,
            )
            z = layers.Dense(
                target_shape[-1], 
                dtype=self.dtype_policy,
                name=name+"/z"
            )(z) if latent_dim_ratio != 1 else z

            reshaper = models.Model(
                inputs, 
                [z, z_mean, z_log_var], 
                name=name+"_"+reshape_type+"_z"
            )
        # Use a scalar batch-size placeholder when no variational statistics exist.
        else:
            dummy_outputs = tf.shape(inputs)[0]

            reshaper = models.Model(
                inputs, 
                [x, dummy_outputs, dummy_outputs], 
                name=name+"_"+reshape_type
            )

        return reshaper

    def _create_token_regularizer(
        self, 
        i: int, 
        layers_dicts: list[dict], 
        layers_dict: dict, 
        base_dim: int, 
        kwargs: dict = {}, 
        name: str | None = None
    ) -> layers.Layer:
        """Create an auxiliary class-token softmax head.

        Args:
            i (int): Zero-based current stage index.
            layers_dicts (list[dict]): Previously constructed stages.
            layers_dict (dict): Components already created at this stage.
            base_dim (int): Depth-zero feature width used for inference.
            kwargs (dict): Token-slice bounds and optional MLP settings.
            name (str | None): Keras layer name.

        Returns:
            tf.keras.layers.Layer: A direct Dense softmax, or a two-Dense
            Sequential head when ``mlp_ratio`` is configured.
        """

        input_dim = self._get_current_output_dim(
            i, 
            layers_dicts, 
            layers_dict, 
            base_dim
        ) 
        input_dim *= kwargs["end"] - kwargs["start"]

        mlp_ratio = kwargs.get("mlp_ratio", None)
        activation_function = kwargs.get(
            "activation_function", "tanh"
        )

        # Preserve the original one-layer topology when no MLP is requested.
        if mlp_ratio is None:
            return layers.Dense(
                self.num_classes,
                activation="softmax",
                dtype=self.dtype_policy.variable_dtype,
                name=name
            )

        token_regularizer = models.Sequential([
            layers.Dense(
                max(1, int(input_dim * mlp_ratio)), 
                activation=activation_function, 
                dtype=self.dtype_policy,
                name=f"{name}/first_layer"
            ), 
            layers.Dense(
                self.num_classes, 
                activation="softmax", 
                dtype=self.dtype_policy.variable_dtype,
                name=f"{name}/final_layer"
            )
        ], name=name)

        return token_regularizer

    def _create_layer_dict(
        self, 
        i: int, 
        layers_dicts: list[dict]
    ) -> dict[str, layers.Layer]:
        """Construct all components configured for one transformer depth.

        Args:
            i (int): Zero-based stage index; the public depth key is ``i + 1``.
            layers_dicts (list[dict]): Previously constructed stages used for
                dimension and grid inference.

        Returns:
            dict[str, tf.keras.layers.Layer | tf.keras.Model]: Ordered mapping
            containing only components selected for this depth.  Components run
            in connector, cross-attention, block, mixer, downsample, upsample,
            reshape, regularizer order.
        """

        layers_dict = {}
        key = i+1

        # Build this depth's routed feature connector when configured.
        if key in self.connection_ids_dict:
            layers_dict[self.FC] = self._create_feature_handler(
                ids_set=self.connection_ids_dict[key], 
                layers_dicts=layers_dicts, 
                base_dim=self.dim, 
                dim_forced=self.dim_forced, 
                ln_mlp_ratio=self.ln_mlp_ratio, 
                ln_no_adaptation=self.ln_no_adaptation, 
                kwargs=self.connection_kwargs, 
                name=f"{self.name_prefix}depth_{key}_{self.FC[2:]}"
            )
        
        # Build this depth's cross-attention feature handler when configured.
        if key in self.cross_attention_ids_dict:
            layers_dict[self.CAC] = self._create_feature_handler(
                ids_set=self.cross_attention_ids_dict[key], 
                layers_dicts=layers_dicts, 
                base_dim=self.dim, 
                dim_forced=self.dim_forced, 
                ln_mlp_ratio=self.ln_mlp_ratio, 
                ln_no_adaptation=self.ln_no_adaptation, 
                kwargs=self.cross_attention_kwargs, 
                name=f"{self.name_prefix}depth_{key}_{self.CAC[2:]}"
            )

        # Build a transformer block with the width produced by preceding handlers.
        if key in self.vit_block_ids:
            mha_query_dim = layers_dict[self.CAC].output_dim if \
                            key in self.cross_attention_ids_dict and \
                            self.cross_attention_plug_type == "queries" \
                            else None

            layers_dict[self.VTB] = self._create_vit_block(
                i=i, layers_dicts=layers_dicts, 
                layers_dict=layers_dict, base_dim=self.dim, 
                mha_key_dim=self.mha_key_dim, 
                mha_value_dim=self.mha_value_dim, 
                mha_query_dim=mha_query_dim, 
                mha_num_heads=self.mha_num_heads, 
                mlp_ratio=self.vit_block_mlp_ratio, 
                mlp_output_dim=self.vit_block_mlp_output_dims.get(key, None), 
                ln_mlp_ratio=self.ln_mlp_ratio, 
                ln_no_adaptation=self.ln_no_adaptation, 
                drop_prob=self.drop_prob, 
                drop_per_sample=self.drop_per_sample, 
                use_decoder=key in self.use_decoder_ids, 
                name_prefix=f"{self.name_prefix}depth_{key}_"
            )

        # Build this depth's convolutional local mixer when configured.
        if key in self.local_mixer_ids:
            layers_dict[self.LM] = self._create_local_mixer(
                i=i, 
                dim_forced=self.dim_forced, 
                layers_dicts=layers_dicts, 
                layers_dict=layers_dict, 
                base_dim=self.dim, 
                base_grid_size=self.grid_size, 
                ln_mlp_ratio=self.ln_mlp_ratio, 
                ln_no_adaptation=self.ln_no_adaptation, 
                circumvent_tokens=int(self.cls_token_type is not None) + 
                                int(self.distil_token_type is not None), 
                kwargs=self.local_mixer_kwargs, 
                name=f"{self.name_prefix}depth_{key}_{self.LM[2:]}"
            )

        # Build this depth's downsampler when configured.
        if key in self.downsample_ids:
            layers_dict[self.DS] = self._create_scaler(
                scaler_type="downsample", 
                i=i, 
                dim_forced=self.dim_forced, 
                layers_dicts=layers_dicts, 
                layers_dict=layers_dict, 
                base_dim=self.dim, 
                base_grid_size=self.grid_size, 
                ln_mlp_ratio=self.ln_mlp_ratio, 
                ln_no_adaptation=self.ln_no_adaptation, 
                circumvent_tokens=int(self.cls_token_type is not None) + 
                                int(self.distil_token_type is not None), 
                kwargs=self.downsample_kwargs, 
                name=f"{self.name_prefix}depth_{key}_{self.DS[2:]}"
            )

        # Build this depth's upsampler when configured.
        if key in self.upsample_ids:
            layers_dict[self.US] = self._create_scaler(
                scaler_type="upsample", 
                i=i, 
                dim_forced=self.dim_forced, 
                layers_dicts=layers_dicts, 
                layers_dict=layers_dict, 
                base_dim=self.dim, 
                base_grid_size=self.grid_size, 
                ln_mlp_ratio=self.ln_mlp_ratio, 
                ln_no_adaptation=self.ln_no_adaptation, 
                circumvent_tokens=int(self.cls_token_type is not None) + 
                                int(self.distil_token_type is not None), 
                kwargs=self.upsample_kwargs, 
                name=f"{self.name_prefix}depth_{key}_{self.US[2:]}"
            )

        # Build this depth's flatten or unflatten reshaper when configured.
        if key in self.reshaper_ids_dict:
            layers_dict[self.R] = self._create_reshaper(
                reshape_type=self.reshaper_ids_dict[key], 
                i=i, layers_dicts=layers_dicts, 
                layers_dict=layers_dict, base_dim=self.dim, 
                base_grid_size=self.grid_size, 
                grid_has_tokens=int(self.cls_token_type is not None) + 
                                int(self.distil_token_type is not None), 
                kwargs=self.reshaper_kwargs, 
                name=f"{self.name_prefix}depth_{key}_{self.R[2:]}"
            )

        # Build this depth's auxiliary token classifier when configured.
        if key in self.cls_token_regularizer_ids:
            layers_dict[self.CTR] = self._create_token_regularizer(
                i=i, 
                layers_dicts=layers_dicts, 
                layers_dict=layers_dict, 
                base_dim=self.dim, 
                kwargs=self.cls_token_regularizer_kwargs, 
                name=f"{self.name_prefix}depth_{key}_{self.CTR[2:]}"
            )

        return layers_dict

    def _create_layers(self) -> None:
        """Create the ``depth`` stage dictionaries.

        Returns:
            None: ``layers_dicts`` is replaced by a list of length ``depth``;
            with ``depth=0`` it is an empty list.
        """

        self.layers_dicts = []

        for i in range(self.depth):
            self.layers_dicts.append(
                self._create_layer_dict(i, self.layers_dicts)
            )

    def _create_unpatchifier(self) -> None:
        """Create the final adaptive projection and image reconstruction head.

        The head removes no class token itself; :meth:`encode` does so first.
        It normalizes final tokens with the condition, projects each token to a
        flattened patch, rearranges patches into ``[B, H, W, channels]``, and
        optionally applies the residual CNN refiner.

        Returns:
            None: ``self.unpatchifier`` is assigned when
            ``use_unpatchify=True``.  No attribute is created otherwise.
        """

        # Derive the final token width required by image unpatchification.
        if self.use_unpatchify:
            dim = self._get_unforced_total_dim(
                [self.depth], 
                self.layers_dicts, 
                self.dim
            )

            name = f"{self.name_prefix}unpatchifier"

            token_inputs = layers.Input(
                shape=(None, dim), # (grid_size * grid_size, dim)
                dtype=self.compute_dtype, 
                name=name+"token_inputs"
            )
            cond_inputs = layers.Input(
                shape=(self.cond_dim,), 
                dtype=self.compute_dtype, 
                name=name+"cond_inputs"
            )

            x = AdaLNZero(
                dim=dim, 
                return_gate=False, 
                mlp_ratio=self.ln_mlp_ratio, 
                no_adaptation=self.ln_no_adaptation, 
                name=f"{name}/layer_norm"
            )((token_inputs, cond_inputs))
            x = layers.Dense(
                self.patch_size * self.patch_size * self.channels, 
                kernel_initializer="zeros", 
                activation=self.final_ffn_activation_func, 
                name=f"{name}/ffn"
            )(x)

            x_shape = tf.shape(x)
            batch_size = x_shape[0]
            grid_size = tf.cast(
                tf.sqrt(
                    tf.cast(x_shape[1], self.dtype_policy.variable_dtype)
                ), 
                tf.int32
            )

            x = tf.reshape(x, (
                batch_size, 
                grid_size, # self._current_resolution // self.patch_size, 
                grid_size, # self._current_resolution // self.patch_size, 
                self.patch_size, 
                self.patch_size, 
                self.channels
            ), name=f"{name}/reshape_1")
            x = tf.transpose(x, 
                perm=(0, 1, 3, 2, 4, 5), 
                name=f"{name}/transpose"
            )
            x = tf.reshape(x, (
                batch_size, 
                grid_size * self.patch_size, # self._current_resolution, 
                grid_size * self.patch_size, # self._current_resolution, 
                self.channels
            ), name=f"{name}/reshape_2")

            # Optionally refine the unpatchified image with residual convolutions.
            if self.use_refiner_cnn:
                hidden_dim = dim if self.refiner_cnn_hidden_dim is None \
                            else self.refiner_cnn_hidden_dim

                h = layers.Conv2D(
                    hidden_dim, 
                    kernel_size=3, 
                    padding="same", 
                    activation="swish", 
                    name=f"{name}/refiner_conv_1"
                )(x)
                h = layers.Conv2D(
                    self.channels, 
                    kernel_size=3, 
                    padding="same", 
                    kernel_initializer="zeros", 
                    bias_initializer="zeros", 
                    name=f"{name}/refiner_conv_2"
                )(h)
                x = x + h if self.refiner_cnn_residual else h 

            outputs = layers.Activation(
                self.final_activation_func, 
                name=f"{name}/noises"
            )(x)

            self.unpatchifier = models.Model(
                inputs=[token_inputs, cond_inputs], 
                outputs=outputs, 
                name=name
            )

    def _build_model(self, call_model: bool = True) -> list[tf.TensorShape]:
        """Create symbolic Keras inputs for the active resolution.

        Args:
            call_model (bool): Execute :meth:`call` on the symbolic inputs and
                assign ``self.outputs`` when true; otherwise leave outputs None.

        Returns:
            list[tf.TensorShape]: Shapes for image ``[None, H, H, C]``, scalar
            timestep ``[None]``, and scalar label ``[None]`` inputs.  The
            image dtype follows the policy compute dtype; timestep is
            ``tf.int32`` and label is ``tf.uint8``.
        """

        noisy_images = layers.Input(
            shape=(
                self._current_resolution, 
                self._current_resolution, 
                self.channels
            ), 
            dtype=self.compute_dtype,
            name="noisy_images"
        )
        ts = layers.Input(
            shape=(), 
            dtype=tf.int32, 
            name="timesteps"
        )
        labels = layers.Input(
            shape=(), 
            dtype=tf.uint8, 
            name="labels"
        )

        self.inputs = (noisy_images, ts, labels)
        self.outputs = self.call(
            self.inputs
        ) if call_model else None

        input_shape = [
            input_layer.shape 
            for input_layer in self.inputs
        ]

        return input_shape

    def _expand_token_regularizer(
        self, 
        regularizer: layers.Layer | None, 
        source_regularizer: layers.Layer | None = None
    ) -> layers.Layer | None:
        """Append one output to an auxiliary softmax while preserving weights.

        Args:
            regularizer (tf.keras.layers.Layer | None): Head to expand.
            source_regularizer (tf.keras.layers.Layer | None): Optional
                expanded raw head supplying the new EMA output parameters.

        Returns:
            tf.keras.layers.Layer | None: Expanded head, or ``None`` when the
            input head is disabled.
        """

        # Leave disabled regularizers untouched.
        if regularizer is None:
            return None

        old_layer = regularizer.layers[-1] if isinstance(regularizer, models.Sequential) \
                    else regularizer
        old_kernel, old_bias = old_layer.get_weights()
        layer_config = old_layer.get_config()
        layer_config["units"] = self.num_classes

        new_layer = old_layer.__class__.from_config(layer_config)
        new_layer(
            tf.zeros((1, old_kernel.shape[0]), dtype=old_kernel.dtype), 
            training=False
        )
        new_kernel, new_bias = new_layer.get_weights()
        new_kernel[..., :-1] = old_kernel
        new_bias[:-1] = old_bias

        # Initialize only the new EMA output from the expanded raw head.
        if source_regularizer is not None:
            source_layer = source_regularizer.layers[-1] \
                        if isinstance(source_regularizer, models.Sequential) \
                        else source_regularizer
            source_kernel, source_bias = source_layer.get_weights()
            new_kernel[..., -1] = source_kernel[..., -1]
            new_bias[-1] = source_bias[-1]

        new_layer.set_weights([new_kernel, new_bias])

        # Retain an optional hidden layer and replace only its final softmax.
        if isinstance(regularizer, models.Sequential):
            regularizer.pop()
            regularizer.add(new_layer)
            return regularizer

        return new_layer

    @property
    def current_resolution(self) -> int:
        """Return the square image resolution currently processed.

        Returns:
            int: Active positive integer resolution.
        """

        return self._current_resolution

    def build(
        self, 
        input_shape: tuple[tuple, tuple, tuple] | None = None
    ) -> None:
        """Build the network against its current configured resolution.

        Args:
            input_shape (tuple[tuple, tuple, tuple] | None): Accepted for the
                Keras ``Model.build`` protocol but ignored; symbolic shapes are
                generated by :meth:`_build_model` from model configuration.

        Returns:
            None: Variables are created and the Keras built flag is set.
        """

        del input_shape
        configured_shapes = self._build_model(call_model=True)

        # Mark the Keras model built only after its symbolic graph exists.
        if not self.built:
            super().build(configured_shapes)

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        full_return: bool = False, 
        min_depth: int = 0, 
        training: bool | None = None
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor, list[tf.Tensor], 
        list[tf.Tensor], list[tuple[tf.Tensor, tf.Tensor]]]:
        """Run embedding, configured depths, and the optional output head.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): ``(images, times,
                labels)``.  At ``min_depth=0``, images are float tensors
                ``[B, H, W, channels]``, times are integer ``[B]``, and labels
                are integer ``[B]``.  At ``min_depth>0``, the first tensor is
                the already encoded representation required at that depth.
            full_return (bool): Return intermediate conditioning, features,
                regularizer predictions, and VAE statistics when true.
            min_depth (int): First depth to execute.  ``0`` performs patch/input
                embedding; ``k`` in ``1..depth`` treats ``inputs[0]`` as depth-k
                input and skips stages before k.  This is used by VAE decoding.
            training (bool | None): Keras training mode passed to every layer.

        Returns:
            tf.Tensor | tuple: Normally an image/noise tensor ``[B, H, W, C]``
            (or final rank-3 tokens when ``use_unpatchify=False``).  With
            ``full_return=True``, returns ``(output, cond, features_list,
            regs_list, z_vals_list)``. Each ``z_vals_list`` item is one
            ``(mean, log_variance)`` pair. Feature index 0 is depth 0 and
            index k is depth k; absent regularizers are ``None``.
        """

        x, cond, features_list, regs_list, z_vals_list = self.encode(
            inputs, 
            min_depth=min_depth, 
            training=training
        )
        noises = self.unpatchifier(
            (x, cond), 
            training=training
        ) if self.use_unpatchify else x

        # Include condition, features, regularizers, and latent values only on request.
        if full_return:
            return noises, cond, features_list, regs_list, z_vals_list 
        return noises

    def set_current_resolution(self, resolution: int | None = None) -> None:
        """Set the active square resolution used by embeddings and reshapers.

        Args:
            resolution (int | None): Positive size divisible by ``patch_size``.
                ``None`` restores the constructor ``image_size``.  Values may
                differ from and exceed the base image size.

        Returns:
            None: ``current_resolution`` changes in place.

        Raises:
            AssertionError: If the value is nonintegral, nonpositive, or not
                divisible by ``patch_size``.
        """

        resolution = self.image_size if resolution is None else resolution

        require(
            not isinstance(resolution, bool) and int(resolution) == resolution, 
            "resolution must be an integer."
        )
        require(
            resolution > 0, 
            "resolution must be positive."
        )
        require(
            resolution % self.patch_size == 0, 
            "resolution must be divisible by patch_size."
        )


        self._current_resolution = int(resolution)

    def embed_conditions(
        self, 
        times: tf.Tensor, 
        labels: tf.Tensor, 
        cond_type: CondType | None, 
        full_return: bool = False, 
        training: bool | None = None
    ) -> tf.Tensor | None | tuple[
        tf.Tensor | None, 
        tf.Tensor | None, 
        tf.Tensor | None
    ]:
        """Embed the requested subset of timestep and label conditions.

        Args:
            times (tf.Tensor): Integer timestep IDs of shape ``[B]``.
            labels (tf.Tensor): Integer label IDs of shape ``[B]``.
            cond_type (CondType | None): ``"time_label"`` uses both,
                ``"time"`` or ``"label"`` uses one, and ``None`` uses neither.
            full_return (bool): Also return the individual embeddings.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor | None | tuple[tf.Tensor | None, tf.Tensor | None,
            tf.Tensor | None]: Combined condition ``[B, cond_dim]`` (or None),
            optionally as ``(combined, time_embedding, label_embedding)``.
            ``"add"`` preserves per-embedder width; ``"concat"`` appends it.
        """

        cond_type = [] if cond_type is None else cond_type

        time_embeds = self.time_embedder(
            times, 
            training=training
        ) if self.time_embedder is not None and "time" in cond_type else None

        label_embeds = self.label_embedder(
            labels, 
            training=training
        ) if self.label_embedder is not None and (
            "label" in cond_type or 0 in self.cls_token_regularizer_ids
        ) else None

        conds = self.conds_merger(
            (time_embeds, label_embeds), 
            training=training
        ) if self.time_embedder is not None and self.label_embedder is not None \
            and "time" in cond_type and "label" in cond_type else None

        # Derive the combined condition from the configured condition type.
        if conds is None:
            conds = time_embeds if "time" in cond_type else (
                    label_embeds if "label" in cond_type else None
            )

        # Expose component embeddings only for callers requesting full metadata.
        if full_return:
            return conds, time_embeds, label_embeds
        return conds

    def embed_inputs(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        cond_type: CondType, 
        full_return: bool = False, 
        training: bool | None = None
    ) -> tuple[tf.Tensor, object]:
        """Create depth-0 patch tokens and condition embeddings.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Images
                ``[B, H, W, C]``, integer times ``[B]``, and integer labels
                ``[B]``.
            cond_type (CondType | None): Condition subset passed to
                :meth:`embed_conditions`.
            full_return (bool): Request the three-part condition tuple instead
                of only its merged tensor.
            training (bool | None): Keras training mode.

        Returns:
            tuple[tf.Tensor, object]: Patch tokens of shape
            ``[B, (H/patch_size)^2, dim]`` (possibly wider after concatenation)
            and either the merged condition or its full-return tuple.  If
            ``patches_conds_merger_type`` is set, the merged condition is
            repeated across and merged into every patch token.
        """

        images, times, labels = inputs

        conds_list = self.embed_conditions(
            times, labels, cond_type, 
            full_return=full_return, 
            training=training
        )
        conds_merged = conds_list[0] if full_return else conds_list

        x = self.patch_embedder(
            images, 
            output_grid_size=self._current_resolution // self.patch_size if \
                            self.image_size != self._current_resolution \
                            else None, 
            training=training
        )
        x = self.patches_conds_merger((
            x, tf.repeat(
                conds_merged[:, None], 
                tf.shape(x)[1], 
                axis=1
            )), training=training
        ) if self.patches_conds_merger is not None else x

        return x, conds_list

    def prepend_single_token(
        self, 
        x: tf.Tensor, 
        token: SingleTokenLayer, 
        token_type: TokenType, 
        time_embeds: tf.Tensor | None = None, 
        label_embeds: tf.Tensor | None = None, 
        times: tf.Tensor | None = None, 
        labels: tf.Tensor | None = None, 
        training: bool | None = None
    ) -> tf.Tensor:
        """Prepend the configured learned or condition-derived single token.

        Args:
            x (tf.Tensor): Patch tokens ``[B, P, D]``.
            token (SingleTokenLayer): Layer that creates the prefix token.
            token_type (TokenType): ``"new_weight"``, ``"time"``,
                ``"label"``, or ``"time_label"``.
            time_embeds (tf.Tensor | None): Reusable ``[B, E]`` time embedding;
                when omitted for a time token, it is computed from ``times``.
            label_embeds (tf.Tensor | None): Reusable label embedding.
            times (tf.Tensor | None): Integer ``[B]`` IDs required when a time
                embedding was not supplied.
            labels (tf.Tensor | None): Integer ``[B]`` IDs required when a label
                embedding was not supplied.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor: Token sequence ``[B, P + 1, D_out]`` with the new token
            at index 0.
        """

        # Leave the patch sequence unchanged when this prefix token is disabled.
        if token_type is None:
            return x

        # Initialize the class token from timestep embeddings.
        if token_type == "time":
            embeds = self.time_embedder(
                times, 
                training=training
            ) if time_embeds is None else time_embeds
        # Initialize the class token from label embeddings.
        elif token_type == "label":
            embeds = self.label_embedder(
                labels, 
                training=training
            ) if label_embeds is None else label_embeds
        # Initialize the class token from merged time and label conditions.
        elif token_type == "time_label":
            embeds = self.embed_conditions(
                times, labels, 
                token_type, 
                full_return=False, 
                training=training
            )
        # Use the learned standalone token instead of per-example embeddings.
        elif token_type == "new_weight":
            embeds = None

        x = tf.concat([
            token(
                (x, embeds), 
                training=training
            ), 
            x
        ], axis=1)

        return x

    def prepend_cls_token(
        self,
        x: tf.Tensor,
        cls_token_type: TokenType,
        time_embeds: tf.Tensor | None = None,
        label_embeds: tf.Tensor | None = None,
        times: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        training: bool | None = None
    ) -> tf.Tensor:
        """Prepend the configured class token.

        This public helper keeps the class-token API while delegating to the
        shared single-token implementation.

        Args:
            x (tf.Tensor): Patch tokens ``[B, P, D]``.
            cls_token_type (TokenType): Class-token source.
            time_embeds (tf.Tensor | None): Reusable time embedding.
            label_embeds (tf.Tensor | None): Reusable label embedding.
            times (tf.Tensor | None): Integer timestep IDs.
            labels (tf.Tensor | None): Integer class IDs.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor: Sequence with the class token prepended.
        """

        return self.prepend_single_token(
            x, self.cls_token, cls_token_type,
            time_embeds=time_embeds,
            label_embeds=label_embeds,
            times=times,
            labels=labels,
            training=training
        )

    def prepend_distil_token(
        self,
        x: tf.Tensor,
        distil_token_type: TokenType,
        time_embeds: tf.Tensor | None = None,
        label_embeds: tf.Tensor | None = None,
        times: tf.Tensor | None = None,
        labels: tf.Tensor | None = None,
        training: bool | None = None
    ) -> tf.Tensor:
        """Prepend the configured distillation token.

        Args:
            x (tf.Tensor): Patch tokens ``[B, P, D]``.
            distil_token_type (TokenType): Distillation-token source.
            time_embeds (tf.Tensor | None): Reusable time embedding.
            label_embeds (tf.Tensor | None): Reusable label embedding.
            times (tf.Tensor | None): Integer timestep IDs.
            labels (tf.Tensor | None): Integer class IDs.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor: Sequence with the distillation token prepended.
        """

        return self.prepend_single_token(
            x, self.distil_token, distil_token_type,
            time_embeds=time_embeds,
            label_embeds=label_embeds,
            times=times,
            labels=labels,
            training=training
        )

    def slice_and_flatten_tokens(
        self, 
        x: tf.Tensor, 
        start: int, 
        end: int
    ) -> tf.Tensor:
        """Select a token interval and flatten it per sample.

        Args:
            x (tf.Tensor): Usually rank-3 ``[B, tokens, D]``; rank-2 inputs are
                assumed already flattened.
            start (int): Inclusive Python token-slice bound.
            end (int): Exclusive bound.  The final reshape assumes ``end-start``
                selected tokens.

        Returns:
            tf.Tensor: Rank-2 tensor. Rank-2 inputs are returned unchanged;
            rank-3 inputs have their selected token interval flattened.
        """

        # A variational flatten reshaper has already combined token and channel
        # axes, so there is no token interval left to slice.
        if x.shape.rank == 2:
            return x

        x = x[:, start: end, :]
        x_shape = tf.shape(x)
        x = tf.reshape(x, (
            x_shape[0],
            x_shape[-1] * (end-start),
        ))

        return x

    def encode(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        max_depth: int = -1, 
        min_depth: int = 0, 
        training: bool | None = None
    ) -> tuple[tf.Tensor, tf.Tensor, list[tf.Tensor], 
        list[tf.Tensor], list[tuple[tf.Tensor, tf.Tensor]]]:
        """Encode inputs through a selectable contiguous range of depths.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): At ``min_depth=0``,
                image ``[B,H,W,C]``, time IDs ``[B]``, and label IDs ``[B]``.
                At ``min_depth>0``, the first item is an already embedded token
                or flattened latent tensor positioned at that depth.
            max_depth (int): Exclusive zero-based loop stop.  ``-1`` (default)
                does not stop early; ``0`` executes no stage, and ``k>0``
                executes stages with zero-based indices below k.
            min_depth (int): Number of initial stages to skip, in ``0..depth``.
                Skipped feature slots are filled with ``None``.  ``0`` performs
                depth-0 embedding; values 1..N are intended for resuming from a
                matching flattened/unflattened bottleneck representation.
            training (bool | None): Keras training mode.

        Returns:
            tuple: ``(tokens, cond, features_list, regs_list, z_vals_list)``. Tokens
            exclude optional class and distillation tokens at return, while
            ``features_list`` retains them in class, distillation, patch order.
            ``features_list[k]`` is depth k, ``regs_list[k]`` is its auxiliary
            class distribution or ``None``. ``z_vals_list`` keeps each KL
            flatten reshaper's mean/log-variance pair in execution order.

        Raises:
            AssertionError: If ``min_depth`` is outside ``0..depth``.
        """

        require(
            0 <= min_depth <= self.depth, 
            "min_depth must be in the range of [0, depth]."
        )

        # Embed raw inputs when execution starts at the network entrance.
        if min_depth == 0:
            x, (cond, time_embeds, label_embeds) = self.embed_inputs(
                inputs, 
                self.cond_type, 
                full_return=True, 
                training=training
            )
            x = self.prepend_distil_token(
                x, 
                self.distil_token_type, 
                time_embeds=time_embeds, 
                label_embeds=label_embeds, 
                times=inputs[1], 
                labels=inputs[2], 
                training=training
            )
            x = self.prepend_cls_token(
                x, 
                self.cls_token_type, 
                time_embeds=time_embeds, 
                label_embeds=label_embeds, 
                times=inputs[1], 
                labels=inputs[2], 
                training=training
            )
        # Resume from a precomputed feature while rebuilding its conditions.
        else:
            latent_inputs = list(inputs[0]) if isinstance(
                inputs[0], (list, tuple)
            ) else [inputs[0]]
            x = latent_inputs[0]
            cond, time_embeds, label_embeds = self.embed_conditions(
                inputs[1], inputs[2], 
                self.cond_type, 
                full_return=True, 
                training=training
            )

        z = self.labels_embed_reg(
            label_embeds, 
            training=training
        ) if self.labels_embed_reg is not None else None

        features_list = [None] * min_depth + [x]
        regs_list = [z] + [None] * min_depth
        z_vals_list = []
        latent_index = 1
        for i, layers_dict in enumerate(self.layers_dicts):
            # Stop before the exclusive maximum depth.
            if i == max_depth:
                break
            # Leave earlier stages untouched during resumed execution.
            if i < min_depth:
                continue

            x = layers_dict[self.FC](
                features_list, 
                cond=cond, 
                training=training
            ) if self.FC in layers_dict else x

            h = layers_dict[self.CAC](
                features_list, 
                cond=cond, 
                training=training
            ) if self.CAC in layers_dict else None

            x = layers_dict[self.VTB](
                (x, cond), 
                queries=h if self.cross_attention_plug_type == "queries" else None, 
                values=h if self.cross_attention_plug_type == "values" else None, 
                training=training
            ) if self.VTB in layers_dict else x

            x = layers_dict[self.LM](
                (x, cond), 
                training=training
            ) if self.LM in layers_dict else x

            x = layers_dict[self.DS](
                (x, cond), 
                training=training
            ) if self.DS in layers_dict else x

            x = layers_dict[self.US](
                (x, cond), 
                training=training
            ) if self.US in layers_dict else x

            is_flatten = self.reshaper_ids_dict.get(i + 1) == "flatten"
            if self.R in layers_dict and min_depth > 0 and is_flatten:
                x = latent_inputs[latent_index]
                latent_index += 1
                x_mean, x_log_var = None, None
            else:
                x, x_mean, x_log_var = layers_dict[self.R](
                    x, 
                    training=training
                ) if self.R in layers_dict else (x, None, None)

            z = layers_dict[self.CTR](
                self.slice_and_flatten_tokens(
                    x, 
                    self.cls_token_regularizer_kwargs["start"], 
                    self.cls_token_regularizer_kwargs["end"]
                ), 
                training=training
            ) if self.CTR in layers_dict else None

            features_list.append(x)
            regs_list.append(z)
            if x_mean is not None and is_flatten and bool(
                self.reshaper_kwargs.get("add_kl", False)
            ):
                z_vals_list.append((x_mean, x_log_var))

        prefix_tokens_num = int(self.cls_token_type is not None) + \
                            int(self.distil_token_type is not None)
        x = x[:, prefix_tokens_num:] if prefix_tokens_num else x

        return x, cond, features_list, regs_list, z_vals_list

    def get_variables_names(
        self, 
        vars: list[tf.Variable] | None = None
    ) -> list[str]:
        """Return TensorFlow names for selected trainable variables.

        Args:
            vars (list[tf.Variable] | None): Variables to inspect.  ``None``
                selects every current trainable variable.

        Returns:
            list[str]: Variable names in input/model order.
        """

        vars = self.trainable_variables if vars is None else vars
        names = [var.name for var in vars]

        return names

    def add_depths(
        self, 
        depth_spec: str | tuple | set | dict | list | None
    ) -> dict[str, dict[str, int]]:
        """Append transformer depths with the existing layer factories.

        This method is the structural part of progressive-depth training. A
        string adds one depth containing that layer. A tuple or set combines
        several layer types in one depth, while an outer list adds one depth
        for each item. A dictionary also describes one depth and may provide
        ``ids`` for ``feature_connector`` or ``cross_attention_connector``,
        ``use_decoder`` and ``mlp_output_dim`` for
        ``vision_transformer_block``, or ``reshape_type`` for ``reshaper``.
        Other names are ``local_mixer``, ``downsampler``, ``upsampler``, and
        ``cls_token_regularizer``.  These are the suffixes of the class's layer
        constants and therefore the exact accepted strings.

        The method reuses the model-wide kwargs and the normal ID assertions
        and handlers used at construction. New depths are permanent and their
        constructor metadata is updated so cloning and saving reproduce the
        expanded network. Because the output head already exists, the added
        sequence must finish with the same feature dimension and grid size.

        Args:
            depth_spec (str | tuple | set | dict | list | None): One depth
                specification, a list of specifications, or ``None``. ``None``
                and an empty list leave the network intact.  Example:
                ``[{"feature_connector": {"ids": [-1]}},
                {"vision_transformer_block": {"use_decoder": True}}]``.

        Returns:
            dict[str, dict[str, int]]: ``{"network": {"before": old,
            "added": count, "after": new}}``.

        Raises:
            ValueError: If a layer name is unknown or the appended sequence
                changes the feature width or token grid expected by the
                existing output head.
        """

        depth_specs = depth_spec if isinstance(depth_spec, list) else [depth_spec]
        depth_specs = [spec for spec in depth_specs if spec is not None]
        old_depth = self.depth
        # Leave the architecture unchanged for an empty growth request.
        if len(depth_specs) == 0:
            return {
                "network": {
                    "before": old_depth, 
                    "added": 0, 
                    "after": old_depth
                }
            }

        metadata_names = (
            "connection_ids_dict", "cross_attention_ids_dict", 
            "vit_block_ids", "use_decoder_ids", 
            "vit_block_mlp_output_dims", "local_mixer_ids", 
            "downsample_ids", "upsample_ids", "reshaper_ids_dict", 
            "cls_token_regularizer_ids", 
        )
        metadata = {name: getattr(self, name) for name in metadata_names}
        old_dim = self._get_last_output_dim(
            old_depth-1, self.layers_dicts, self.dim
        )
        old_grid = self._get_last_grid_size(
            old_depth-1, self.layers_dicts, self.grid_size
        )
        planned_layers = list(self.layers_dicts)

        try:
            for layer_spec in depth_specs:
                # Interpret a string as one enabled progressive layer.
                if isinstance(layer_spec, str):
                    layer_spec = {layer_spec: True}
                # Interpret a collection as several enabled layers in one depth.
                elif isinstance(layer_spec, (tuple, set, frozenset)):
                    layer_spec = dict.fromkeys(layer_spec, True)

                key = len(planned_layers) + 1
                for layer_name, options in layer_spec.items():
                    # Ignore explicitly disabled layers.
                    if options is False:
                        continue

                    # Register routed feature IDs for connector layers.
                    if layer_name in (self.FC[2:], self.CAC[2:]):
                        ids = options.get("ids") if isinstance(options, dict) else options
                        ids = [-1] if ids is None or ids is True else ids
                        ids = [ids] if isinstance(ids, int) else list(ids)
                        dict_name = "connection_ids_dict" if layer_name == self.FC[2:] \
                                    else "cross_attention_ids_dict"

                        local_vars = {
                            "depth": key-1, 
                            dict_name: {key: ids}
                        }
                        self._check_dict_assertions(
                            local_vars, 
                            dict_name, 
                            check_items_num=False, 
                            check_keys=False
                        )
                        ids = self._handle_ids(
                            ids, 
                            depth=key-1, 
                            max_id=key-1
                        )

                        setattr(self, dict_name, {
                            **getattr(self, dict_name), 
                            key: ids
                        })
                    # Register transformer-block mode and output-width options.
                    elif layer_name == self.VTB[2:]:
                        block_options = options if isinstance(options, dict) else {}

                        self.vit_block_ids = [
                            *self.vit_block_ids, key
                        ]
                        self.use_decoder_ids = [
                            *self.use_decoder_ids, key
                        ] if block_options.get("use_decoder", False) else self.use_decoder_ids
                        self.vit_block_mlp_output_dims = {
                            **self.vit_block_mlp_output_dims, 
                            key: block_options["mlp_output_dim"], 
                        } if block_options.get("mlp_output_dim") is not None else self.vit_block_mlp_output_dims
                    # Register a local mixer at the new depth.
                    elif layer_name == self.LM[2:]:
                        self.local_mixer_ids = [
                            *self.local_mixer_ids, key
                        ]
                    # Register a downsampler at the new depth.
                    elif layer_name == self.DS[2:]:
                        self.downsample_ids = [
                            *self.downsample_ids, 
                            key
                        ]
                    # Register an upsampler at the new depth.
                    elif layer_name == self.US[2:]:
                        self.upsample_ids = [
                            *self.upsample_ids, 
                            key
                        ]
                    # Register the new depth's flattening direction.
                    elif layer_name == self.R[2:]:
                        reshape_type = options.get("reshape_type") \
                                    if isinstance(options, dict) else options

                        self.reshaper_ids_dict = {
                            **self.reshaper_ids_dict, key: reshape_type
                        }
                    # Register an auxiliary token classifier at the new depth.
                    elif layer_name == self.CTR[2:]:
                        self.cls_token_regularizer_ids = [
                            *self.cls_token_regularizer_ids, key
                        ]
                    # Reject progressive layer names unsupported by this transformer.
                    else:
                        raise ValueError(
                            f"Unknown progressive classifier layer: {layer_name}."
                        )

                layers_dict = self._create_layer_dict(
                    key-1, planned_layers
                )
                planned_layers.append(layers_dict)

            # Keep progressive output width compatible with the existing output head.
            if self._get_last_output_dim(
                len(planned_layers)-1, 
                planned_layers, self.dim
            ) != old_dim:
                raise ValueError(
                    "Added depths must preserve the output-head feature dimension."
                )
            # The existing image head must retain the same reconstruction size.
            if self.use_unpatchify and self._get_last_grid_size(
                len(planned_layers)-1,
                planned_layers,
                self.grid_size,
            ) != old_grid:
                raise ValueError(
                    "Added depths must preserve the output-head token grid."
                )
        except Exception:
            for name, value in metadata.items():
                setattr(self, name, value)
            raise

        self.layers_dicts.extend(
            planned_layers[old_depth:]
        )
        self.depth = len(self.layers_dicts)
        self._save_init_args({
            "depth": self.depth, 
            **{name: getattr(self, name) for name in metadata_names}, 
        })

        return {
            "network": {
                "before": old_depth, 
                "added": self.depth-old_depth, 
                "after": self.depth, 
            }
        }

    def add_class(self, source_network: object | None = None) -> None:
        """Append one class to the label embedding and auxiliary heads.

        Args:
            source_network (object | None): Optional already-expanded raw
                network.  Its new embedding row initializes the corresponding
                row in an EMA clone; existing EMA rows remain unchanged.

        Returns:
            None: ``num_classes``, ``num_labels``, ``label_embedder``, and all
            configured regularizer outputs are updated in place.

        Raises:
            ValueError: If the network was initialized with a fixed class count.
        """

        # Restrict structural growth to the explicit dynamic constructor mode.
        if not self.dynamic_num_classes:
            raise ValueError(
                "add_class requires num_classes=None at initialization."
            )

        old_label_embedder = self.label_embedder
        self.num_classes += 1
        self._init_config["num_classes"] = self.num_classes
        self.num_labels = self.num_classes + int(self.use_cfg)

        # Expand the label embedder when this architecture uses one.
        if old_label_embedder is not None:
            old_weights = old_label_embedder.get_weights()
            label_config = old_label_embedder.get_config()
            label_config["embed_steps"] = self.num_labels

            new_label_embedder = old_label_embedder.__class__.from_config(
                label_config
            )
            new_label_embedder(
                tf.zeros((1,), dtype=tf.int32),
                training=False
            )
            new_weights = new_label_embedder.get_weights()

            new_weights[0][:-1] = old_weights[0]
            for index in range(1, len(old_weights)):
                new_weights[index] = old_weights[index]

            # Initialize only the new EMA row from its raw-network counterpart.
            if source_network is not None:
                new_weights[0][-1] = (
                    source_network.label_embedder.get_weights()[0][-1]
                )

            new_label_embedder.set_weights(new_weights)
            self.label_embedder = new_label_embedder

        self.labels_embed_reg = self._expand_token_regularizer(
            self.labels_embed_reg, 
            source_network.labels_embed_reg if source_network is not None else None
        )
        for index, layers_dict in enumerate(self.layers_dicts):
            # Expand only stages that own an auxiliary class head.
            if self.CTR in layers_dict:
                layers_dict[self.CTR] = self._expand_token_regularizer(
                    layers_dict[self.CTR], 
                    source_network.layers_dicts[index][self.CTR]
                    if source_network is not None else None
                )


def run_self_tests() -> dict[str, str]:
    """Run deterministic, CPU-small integration tests for DiffusionTransformer.

    The checks cover depth-zero and multi-depth execution, every condition and
    class-token mode, additive/concatenative mergers, CNN and linear patching,
    decoder/encoder attention blocks, both cross-attention plug directions,
    routed and relative IDs, spatial mixers/scalers, variational reshaping,
    auxiliary token heads, output alternatives, progressive depth growth,
    configuration reconstruction, resolution changes, and invalid arguments.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"DiffusionTransformer": "passed"}`` after all
        assertions succeed.
    """

    tf.keras.backend.clear_session()
    tf.random.set_seed(101)
    images = tf.reshape(tf.linspace(-1.0, 1.0, 32), (2, 4, 4, 1))
    times = tf.constant([0, 3], dtype=tf.int32)
    labels = tf.constant([1, 2], dtype=tf.uint8)
    inputs = (images, times, labels)
    base = {
        "num_classes": 2, 
        "use_cfg": True, 
        "timesteps": 4, 
        "image_size": 4, 
        "channels": 1, 
        "patch_size": 2, 
        "dim": 4, 
        "mha_num_heads": 1, 
        "vit_block_mlp_ratio": 1.0, 
    }

    depth_zero = DiffusionTransformer(depth=0, **base)
    output, cond, features, regs, z_values = depth_zero(
        inputs, full_return=True, training=False
    )
    assert output.shape == (2, 4, 4, 1)
    assert output.dtype == tf.float32
    assert cond.shape == (2, 4)
    assert len(depth_zero.layers_dicts) == 0
    assert len(features) == len(regs) == 1
    assert z_values == []
    assert depth_zero.current_resolution == 4
    assert depth_zero._build_model(call_model=False) == [
        tf.TensorShape([None, 4, 4, 1]),
        tf.TensorShape([None]),
        tf.TensorShape([None]),
    ]

    for cond_type in (None, "time", "label", "time_label"):
        model = DiffusionTransformer(
            depth=1,
            cond_type=cond_type,
            ln_no_adaptation=cond_type is None,
            use_unpatchify=cond_type is not None,
            **base,
        )
        value, merged, stage_features, stage_regs, _ = model(
            inputs, full_return=True, training=False
        )
        assert value.shape == (
            (2, 4, 4, 1) if cond_type is not None else (2, 4, 4)
        )
        assert len(stage_features) == len(stage_regs) == 2
        assert (merged is None) == (cond_type is None)

    concat_conditions = DiffusionTransformer(
        depth=0, 
        cond_dim=4, 
        conds_merger_type="concat", 
        patches_conds_merger_type="concat", 
        **base
    )
    concat_tokens, concat_cond = concat_conditions.embed_inputs(
        inputs, 
        "time_label", 
        training=False
    )
    assert concat_cond.shape == (2, 4)
    assert concat_tokens.shape == (2, 4, 8)
    additive_patches = DiffusionTransformer(
        depth=0, 
        cond_dim=4, 
        patches_conds_merger_type="add", 
        **base
    )
    assert additive_patches(inputs, training=False).shape == (2, 4, 4, 1)

    for patchify_with_cnn in (False, True):
        for shift_inputs in (False, True):
            model = DiffusionTransformer(
                depth=0, 
                patchify_with_cnn=patchify_with_cnn, 
                shift_inputs=shift_inputs, 
                **base
            )
            tokens, _ = model.embed_inputs(
                inputs, model.cond_type, training=False
            )
            assert tokens.shape == (2, 4, 4)
            # Verify shifted patch ordering in the shift-input test case.
            if shift_inputs:
                tf.debugging.assert_near(
                    tokens[:, 0, :],
                    tf.repeat(tokens[:1, 0, :], 2, axis=0),
                )

    for token_type in ("new_weight", "time", "label", "time_label"):
        for merger_type in ("add", "concat"):
            # Confirm incompatible concatenated class-token widths fail construction.
            if merger_type == "concat" and token_type != "new_weight":
                try:
                    DiffusionTransformer(
                        depth=0, 
                        cls_token_type=token_type, 
                        cls_token_pos_merger_type=merger_type, 
                        **base
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        "Condition-backed concat class tokens must expose their "
                        "documented width mismatch"
                    )
                continue
            model = DiffusionTransformer(
                depth=0, 
                cls_token_type=token_type, 
                cls_token_pos_merger_type=merger_type, 
                **base
            )
            selected_inputs = inputs
            selected_times, selected_labels = selected_inputs[1:]
            tokens, (_, time_embeds, label_embeds) = model.embed_inputs(
                selected_inputs, model.cond_type, 
                full_return=True, training=False
            )
            tokens = model.prepend_cls_token(
                tokens, token_type, 
                time_embeds=time_embeds, 
                label_embeds=label_embeds, 
                times=selected_times, 
                labels=selected_labels, 
                training=False, 
            )
            expected_batch = 2
            assert tokens.shape[0] == expected_batch and tokens.shape[1] == 5
            assert model(selected_inputs, training=False).shape == (
                expected_batch, 4, 4, 1
            )

    configured_token = DiffusionTransformer(
        depth=0, 
        cls_token_type="new_weight", 
        cls_token_freq_dim=2, 
        cls_token_mlp_ratio=2.0, 
        final_ffn_activation_func="tanh", 
        **base,
    )
    configured_token_output = configured_token(inputs, training=False)
    assert configured_token.cls_token_freq_dim == 2
    assert configured_token.cls_token_mlp_ratio == 2.0
    assert configured_token.final_ffn_activation_func == "tanh"
    assert configured_token_output.shape == (2, 4, 4, 1)
    assert bool(tf.reduce_all(tf.abs(configured_token_output) <= 1.0))

    for plug_type in ("values", "queries"):
        routed = DiffusionTransformer(
            depth=2, 
            connection_ids_dict={2: [0, 1]}, 
            connection_kwargs={"connect_type": "concat", "mlp_output_dim": 4}, 
            cross_attention_ids_dict={2: [-2]}, 
            cross_attention_kwargs={"connect_type": "concat", "mlp_output_dim": 4}, 
            cross_attention_plug_type=plug_type, 
            vit_block_ids=[1, 2], 
            use_decoder_ids=[1], 
            **base,
        )
        routed_output = routed(inputs, training=False)
        assert routed_output.shape == (2, 4, 4, 1)
        assert routed.connection_ids_dict == {2: [0, 1]}
        assert routed.cross_attention_ids_dict == {2: [1]}
        assert routed.vit_block_ids == [1, 2]
        assert routed.use_decoder_ids == [1]

    normalized = DiffusionTransformer(
        depth=2, 
        build=False, 
        vit_block_ids=[None], 
        local_mixer_ids=[-1], 
        cls_token_regularizer_ids=[None], 
        connection_ids_dict={2: [None]}, 
        **base,
    )
    assert normalized.vit_block_ids == [1, 2]
    assert normalized.local_mixer_ids == [2]
    assert normalized.cls_token_regularizer_ids == [0, 1, 2]
    assert normalized.connection_ids_dict == {2: [0, 1]}
    assert normalized._handle_ids([-1, -3], depth=2) == [2, 0]

    local = DiffusionTransformer(
        depth=1, 
        vit_block_ids=[], 
        local_mixer_ids=[1], 
        local_mixer_kwargs={
            "kernel_size": 3, 
            "use_pointwise": True, 
            "pointwise_dim_ratio": 1, 
            "pos_embed_type": None, 
        }, 
        **base,
    )
    assert local(inputs, training=False).shape == (2, 4, 4, 1)

    forced_local_position = DiffusionTransformer(
        depth=1,
        vit_block_ids=[],
        local_mixer_ids=[1],
        local_mixer_kwargs={"pos_merger_type": "concat"},
        **base,
    )
    assert forced_local_position.layers_dicts[0][
        forced_local_position.LM
    ].output_dim == 4
    assert forced_local_position(inputs, training=False).shape == (2, 4, 4, 1)

    for method in ("avg_pooling", "max_pooling", "cnn_stride"):
        down = DiffusionTransformer(
            depth=1, 
            vit_block_ids=[], 
            downsample_ids=[1], 
            downsample_kwargs={"scaling_method": method}, 
            use_unpatchify=False, 
            **base, 
        )
        assert down(inputs, training=False).shape[1] == 1
    pooled_ratio = DiffusionTransformer(
        depth=1,
        vit_block_ids=[],
        downsample_ids=[1],
        downsample_kwargs={
            "cnn_dim_ratio": 2,
            "pos_embed_type": None,
        },
        use_unpatchify=False,
        **base,
    )
    assert pooled_ratio.layers_dicts[0][pooled_ratio.DS].mlp is None
    for method in ("cnn_transpose", "interpolate", "cnn_interpolate"):
        up = DiffusionTransformer(
            depth=1, 
            vit_block_ids=[], 
            upsample_ids=[1], 
            upsample_kwargs={"scaling_method": method}, 
            use_unpatchify=False, 
            **base,
        )
        assert up(inputs, training=False).shape[1] == 16

    for add_kl in (False, True):
        bottleneck = DiffusionTransformer(
            depth=2, 
            vit_block_ids=[], 
            reshaper_ids_dict={1: "flatten", 2: "unflatten"}, 
            reshaper_kwargs={"add_kl": add_kl, "latent_dim_ratio": 1.0}, 
            **base, 
        )
        value, _, _, _, latent = bottleneck(inputs, full_return=True, training=False)
        assert value.shape == (2, 4, 4, 1)
        # Validate tensor latent statistics for a KL-enabled reshaper.
        if add_kl:
            assert len(latent) == 1
            assert latent[0][0].shape == latent[0][1].shape == (2, 16)
        # Keep disabled KL metadata empty.
        else:
            assert latent == []

    regularized = DiffusionTransformer(
        depth=1, 
        cls_token_type="new_weight", 
        cls_token_regularizer_ids=[None], 
        cls_token_regularizer_kwargs={
            "start": 0, "end": 1, "mlp_ratio": 2.0
        },
        **base,
    )
    _, _, _, regularizers, _ = regularized(inputs, full_return=True, training=False)
    assert len(regularizers) == 2
    assert all(item.shape == (2, 2) for item in regularizers)
    tf.debugging.assert_near(
        tf.reduce_sum(regularizers[1], axis=-1), tf.ones((2,)), atol=1e-5
    )
    assert isinstance(regularized.labels_embed_reg, models.Sequential)
    assert regularized.labels_embed_reg.layers[0].units == 8
    assert regularized.labels_embed_reg.layers[0].activation.__name__ == "tanh"
    assert (
        regularized.get_config()["cls_token_regularizer_kwargs"]["mlp_ratio"]
        == 2.0
    )
    direct_regularizer = DiffusionTransformer(
        depth=0,
        cls_token_regularizer_ids=[0],
        **base,
    )
    assert isinstance(direct_regularizer.labels_embed_reg, layers.Dense)
    flat = regularized.slice_and_flatten_tokens(tf.ones((2, 3, 4)), 0, 2)
    assert flat.shape == (2, 8)
    already_flat = tf.ones((2, 8))
    assert regularized.slice_and_flatten_tokens(
        already_flat, 0, 2
    ) is already_flat

    for residual in (False, True):
        refined = DiffusionTransformer(
            depth=0, 
            use_refiner_cnn=True, 
            refiner_cnn_hidden_dim=2, 
            refiner_cnn_residual=residual, 
            final_activation_func="sigmoid", 
            **base, 
        )
        refined_output = refined(inputs, training=False)
        assert refined_output.shape == (2, 4, 4, 1)
        assert bool(tf.reduce_all((0.0 <= refined_output) & (refined_output <= 1.0)))
    token_output = DiffusionTransformer(depth=0, use_unpatchify=False, **base)(
        inputs, training=False
    )
    assert token_output.shape == (2, 4, 4)

    resized = DiffusionTransformer(depth=0, **base)
    resized.set_current_resolution(8)
    large_inputs = (
        tf.zeros((1, 8, 8, 1)), 
        tf.zeros((1,), dtype=tf.int32), 
        tf.ones((1,), dtype=tf.uint8), 
    )
    assert resized(large_inputs, training=False).shape == (1, 8, 8, 1)
    resized.set_current_resolution(None)
    assert resized.current_resolution == 4

    progressive = DiffusionTransformer(depth=0, **base)
    assert progressive.add_depths(None)["network"] == {
        "before": 0, "added": 0, "after": 0
    }
    growth = progressive.add_depths([
        "vision_transformer_block", 
        {"feature_connector": {"ids": [-1]}}, 
        ("vision_transformer_block", "cls_token_regularizer"), 
    ])
    assert growth["network"] == {"before": 0, "added": 3, "after": 3}
    assert progressive(inputs, training=False).shape == (2, 4, 4, 1)
    assert progressive.get_variables_names()
    clone = DiffusionTransformer.from_config(progressive.get_config())
    assert clone.depth == 3
    assert clone(inputs, training=False).shape == (2, 4, 4, 1)
    progressive.connection_kwargs["connect_type"] = "add"
    progressive.use_decoder_ids.append(99)
    progressive.cls_token_regularizer_kwargs["start"] = 1
    pristine_defaults = DiffusionTransformer(depth=0, **base)
    assert pristine_defaults.connection_kwargs == {}
    assert pristine_defaults.use_decoder_ids == []
    assert pristine_defaults.cls_token_regularizer_kwargs == {
        "start": 0, "end": 1,
        "train_type": "normal", "distil_type": "hard"
    }

    progressive_cross = DiffusionTransformer(depth=0, **base)
    cross_growth = progressive_cross.add_depths({
        "cross_attention_connector": {"ids": [-1]}, 
        "vision_transformer_block": {
            "use_decoder": True, 
            "mlp_output_dim": 4, 
        }, 
    })
    assert cross_growth["network"] == {"before": 0, "added": 1, "after": 1}
    assert progressive_cross.cross_attention_ids_dict == {1: [0]}
    assert progressive_cross.use_decoder_ids == [1]
    assert progressive_cross.vit_block_mlp_output_dims == {1: 4}
    assert progressive_cross.CAC in progressive_cross.layers_dicts[0]
    assert isinstance(
        progressive_cross.layers_dicts[0][progressive_cross.VTB],
        DiTDecoderBlock,
    )
    assert progressive_cross(inputs, training=False).shape == (2, 4, 4, 1)

    progressive_local = DiffusionTransformer(depth=0, **base)
    assert progressive_local.add_depths("local_mixer")["network"]["added"] == 1
    assert progressive_local.local_mixer_ids == [1]
    assert progressive_local.LM in progressive_local.layers_dicts[0]
    assert progressive_local(inputs, training=False).shape == (2, 4, 4, 1)

    progressive_spatial = DiffusionTransformer(depth=0, **base)
    spatial_growth = progressive_spatial.add_depths([
        "downsampler", 
        "upsampler", 
    ])
    assert spatial_growth["network"] == {"before": 0, "added": 2, "after": 2}
    assert progressive_spatial.downsample_ids == [1]
    assert progressive_spatial.upsample_ids == [2]
    assert progressive_spatial.DS in progressive_spatial.layers_dicts[0]
    assert progressive_spatial.US in progressive_spatial.layers_dicts[1]
    assert progressive_spatial(inputs, training=False).shape == (2, 4, 4, 1)

    progressive_reshape = DiffusionTransformer(depth=0, **base)
    reshape_growth = progressive_reshape.add_depths([
        {"reshaper": {"reshape_type": "flatten"}}, 
        {"reshaper": {"reshape_type": "unflatten"}}, 
    ])
    assert reshape_growth["network"] == {"before": 0, "added": 2, "after": 2}
    assert progressive_reshape.reshaper_ids_dict == {
        1: "flatten", 
        2: "unflatten", 
    }
    assert all(
        progressive_reshape.R in stage
        for stage in progressive_reshape.layers_dicts
    )
    assert progressive_reshape(inputs, training=False).shape == (2, 4, 4, 1)

    progressive_disabled = DiffusionTransformer(depth=0, **base)
    disabled_growth = progressive_disabled.add_depths({
        "vision_transformer_block": False,
    })
    assert disabled_growth["network"] == {"before": 0, "added": 1, "after": 1}
    assert progressive_disabled.layers_dicts[0] == {}
    assert progressive_disabled.vit_block_ids == []
    assert progressive_disabled(inputs, training=False).shape == (2, 4, 4, 1)

    progressive_empty = DiffusionTransformer(depth=0, **base)
    assert progressive_empty.add_depths([])["network"] == {
        "before": 0, 
        "added": 0, 
        "after": 0, 
    }
    collection_growth = progressive_empty.add_depths({
        "local_mixer", 
        "vision_transformer_block", 
    })
    assert collection_growth["network"] == {"before": 0, "added": 1, "after": 1}
    frozen_growth = progressive_empty.add_depths(
        frozenset({"vision_transformer_block"})
    )
    assert frozen_growth["network"] == {"before": 1, "added": 1, "after": 2}
    assert progressive_empty.local_mixer_ids == [1]
    assert progressive_empty.vit_block_ids == [1, 2]
    assert progressive_empty(inputs, training=False).shape == (2, 4, 4, 1)

    progressive_connector_options = DiffusionTransformer(depth=0, **base)
    connector_growth = progressive_connector_options.add_depths([
        {"feature_connector": True}, 
        {"feature_connector": None}, 
        {"feature_connector": 0}, 
        {"cross_attention_connector": True}, 
    ])
    assert connector_growth["network"] == {"before": 0, "added": 4, "after": 4}
    assert progressive_connector_options.connection_ids_dict == {
        1: [0], 
        2: [1], 
        3: [0], 
    }
    assert progressive_connector_options.cross_attention_ids_dict == {4: [3]}
    assert progressive_connector_options(inputs, training=False).shape == (
        2, 4, 4, 1
    )

    progressive_rollback = DiffusionTransformer(depth=0, **base)
    rollback_metadata_names = (
        "connection_ids_dict", 
        "cross_attention_ids_dict", 
        "vit_block_ids", 
        "use_decoder_ids", 
        "vit_block_mlp_output_dims", 
        "local_mixer_ids", 
        "downsample_ids", 
        "upsample_ids", 
        "reshaper_ids_dict", 
        "cls_token_regularizer_ids", 
    )
    rollback_metadata = {
        name: getattr(progressive_rollback, name).copy()
        for name in rollback_metadata_names
    }
    for incompatible_specification in (
        {"vision_transformer_block": {"mlp_output_dim": 6}},
        {"reshaper": {"reshape_type": "flatten"}},
    ):
        try:
            progressive_rollback.add_depths(incompatible_specification)
        except ValueError as error:
            assert "preserve the output-head feature dimension" in str(error)
        else:
            raise AssertionError("Incompatible progressive output width must fail")
        assert progressive_rollback.depth == 0
        assert len(progressive_rollback.layers_dicts) == 0
        assert all(
            getattr(progressive_rollback, name) == rollback_metadata[name]
            for name in rollback_metadata_names
        )
    assert progressive_rollback(inputs, training=False).shape == (2, 4, 4, 1)

    grid_rollback = DiffusionTransformer(depth=0, **base)
    try:
        grid_rollback.add_depths("downsampler")
    except ValueError as error:
        assert "preserve the output-head token grid" in str(error)
    else:
        raise AssertionError("Incompatible progressive output grid must fail")
    assert grid_rollback.depth == 0 and not grid_rollback.layers_dicts

    parser_rollback = DiffusionTransformer(depth=0, **base)
    parser_metadata = {
        name: getattr(parser_rollback, name).copy()
        for name in rollback_metadata_names
    }
    try:
        parser_rollback.add_depths(["local_mixer", "unknown"])
    except ValueError as error:
        assert "Unknown progressive classifier layer" in str(error)
    else:
        raise AssertionError("A later unknown progressive layer must fail")
    assert parser_rollback.depth == 0
    assert len(parser_rollback.layers_dicts) == 0
    assert all(
        getattr(parser_rollback, name) == parser_metadata[name]
        for name in rollback_metadata_names
    )

    positional_modes = (
        "new_weight", "1d_sincos", "1d_interpolate",
        "1d_learned_interpolate", "2d_sincos", 
        "2d_interpolate", "2d_learned_interpolate",
    )
    for positional_mode in positional_modes:
        positional = DiffusionTransformer(
            depth=0, 
            patches_pos_embed_type=positional_mode, 
            **base,
        )
        assert positional(inputs, training=False).shape == (2, 4, 4, 1)
        positional.set_current_resolution(8)
        assert positional(large_inputs, training=False).shape == (1, 8, 8, 1)
    concatenated_position = DiffusionTransformer(
        depth=0, 
        patches_pos_merger_type="concat", 
        **base
    )
    concat_position_tokens, _ = concatenated_position.embed_inputs(
        inputs, concatenated_position.cond_type, training=False
    )
    assert concat_position_tokens.shape == (2, 4, 4)
    assert concatenated_position(inputs, training=False).shape == (2, 4, 4, 1)

    embedding_options = (
        {
            "time_embed_type": "new_weight", 
            "time_freq_dim": 2, 
            "time_embed_trainable": False, 
            "time_mlp_ratio": 2.0, 
            "label_embed_type": "new_weight", 
            "label_freq_dim": 2, 
            "label_embed_trainable": False, 
            "label_mlp_ratio": 2.0, 
        }, 
        {
            "time_embed_type": "1d_sincos", 
            "time_freq_dim": 4, 
            "time_embed_trainable": True, 
            "time_mlp_ratio": 1.0, 
            "label_embed_type": "1d_sincos", 
            "label_freq_dim": 4, 
            "label_embed_trainable": True, 
            "label_mlp_ratio": 1.0, 
        },
    )
    for embed_kwargs in embedding_options:
        embedded = DiffusionTransformer(depth=0, **base, **embed_kwargs)
        embedded_cond, embedded_time, embedded_label = embedded.embed_conditions(
            times, labels, "time_label", full_return=True, training=True
        )
        assert embedded_cond.shape == embedded_time.shape == embedded_label.shape == (
            2, 4
        )
        assert embedded(inputs, training=True).shape == (2, 4, 4, 1)
        assert embedded.time_embedder.embed.trainable is (
            True if embed_kwargs["time_embed_type"] == "new_weight"
            else embed_kwargs["time_embed_trainable"]
        )
        assert embedded.label_embedder.embed.trainable is (
            True if embed_kwargs["label_embed_type"] == "new_weight"
            else embed_kwargs["label_embed_trainable"]
        )

    unforced = DiffusionTransformer(
        depth=2, 
        dim_forced=False, 
        connection_ids_dict={2: [0, 1]}, 
        connection_kwargs={"connect_type": "concat"}, 
        **base,
    )
    assert unforced.layers_dicts[1][unforced.FC].output_dim == 8
    assert unforced(inputs, training=False).shape == (2, 4, 4, 1)
    additive_connection = DiffusionTransformer(
        depth=2, 
        connection_ids_dict={2: [0, 1]}, 
        connection_kwargs={"connect_type": "add"}, 
        **base,
    )
    assert additive_connection(inputs, training=False).shape == (2, 4, 4, 1)

    explicit_attention = DiffusionTransformer(
        depth=1, 
        mha_key_dim=2, 
        mha_value_dim=3, 
        mha_num_heads=2, 
        vit_block_mlp_output_dims={1: 6}, 
        drop_prob=0.5, 
        drop_per_sample=False, 
        **{key: value for key, value in base.items() if key != "mha_num_heads"},
    )
    attention_block = explicit_attention.layers_dicts[0][explicit_attention.VTB]
    assert attention_block.key_dim == 2
    assert attention_block.value_dim == 3
    assert attention_block.mlp_output_dim == 6
    assert attention_block.drop_prob == 0.5
    assert attention_block.drop_per_sample is False
    attention_training = explicit_attention(inputs, training=True)
    attention_evaluation = explicit_attention(inputs, training=False)
    assert attention_training.shape == attention_evaluation.shape == (2, 4, 4, 1)
    assert bool(tf.reduce_all(tf.math.is_finite(attention_training)))

    broad_local = DiffusionTransformer(
        depth=1,
        vit_block_ids=[],
        local_mixer_ids=[1],
        local_mixer_kwargs={
            "embed_temperature": 10.0, 
            "use_layer_norm": True, 
            "ln_mlp_ratio": 1.0, 
            "ln_no_adaptation": False, 
            "kernel_size": 3, 
            "strides": 1, 
            "depth_multiplier": 2, 
            "use_pointwise": False, 
            "zero_init": True, 
            "pos_embed_type": "2d_sincos", 
            "pos_interpolation_method": "bilinear", 
            "pos_merger_type": "add", 
            "mlp_ratio": 1.0, 
            "mlp_activation_func": "gelu", 
            "mlp_output_dim": 4, 
        }, 
        **base, 
    )
    assert broad_local(inputs, training=True).shape == (2, 4, 4, 1)

    resumable = DiffusionTransformer(depth=2, **base)
    full_tokens, full_cond, _, _, _ = resumable.encode(inputs, training=False)
    first_tokens, first_cond, first_features, first_regs, _ = resumable.encode(
        inputs, max_depth=1, training=False
    )
    assert first_tokens.shape == (2, 4, 4)
    assert first_cond.shape == full_cond.shape == (2, 4)
    assert len(first_features) == len(first_regs) == 2
    resumed_tokens, resumed_cond, resumed_features, resumed_regs, _ = resumable.encode(
        (first_features[1], times, labels), 
        min_depth=1, training=False
    )
    tf.debugging.assert_near(resumed_tokens, full_tokens, atol=1e-5)
    tf.debugging.assert_near(resumed_cond, full_cond, atol=1e-5)
    assert resumed_features[0] is None and resumed_features[1] is first_features[1]
    assert len(resumed_regs) == 3
    resumed_output = resumable(
        (first_features[1], times, labels), 
        min_depth=1, 
        training=False
    )
    assert resumed_output.shape == (2, 4, 4, 1)

    policy = DiffusionTransformer(
        depth=0, 
        name="policy_transformer", 
        name_prefix="policy/", 
        dtype="float64", 
        **base
    )
    assert policy.name == "policy_transformer"
    assert policy.dtype_policy.name == "float64"
    assert policy.patch_embedder.name.startswith("policy/")
    assert policy(inputs, training=False).dtype == tf.float32
    policy_config = policy.get_config()
    assert policy_config["name_prefix"] == "policy/"
    assert policy_config["name"] == policy.name
    assert policy_config["dtype"] == "float64"
    policy_clone = DiffusionTransformer.from_config(policy_config)
    assert policy_clone.name == policy.name
    assert policy_clone.dtype_policy.name == "float64"
    assert policy_clone.patch_embedder.name.startswith("policy/")

    invalid_cases = (
        {"image_size": 5}, 
        {"cond_type": None, "ln_no_adaptation": False}, 
        {"cls_token_type": "unknown"}, 
        {"cross_attention_plug_type": "unknown"}, 
        {"connection_ids_dict": {2: [0]}, "depth": 1}, 
        {"connection_ids_dict": {1: [-1]}, "depth": 2},
        {"connection_kwargs": {"unknown": 1}}, 
        {"cross_attention_kwargs": {"unknown": 1}}, 
        {"local_mixer_kwargs": {"unknown": 1}}, 
        {"downsample_kwargs": {"unknown": 1}}, 
        {"upsample_kwargs": {"unknown": 1}}, 
        {"reshaper_kwargs": {"unknown": 1}}, 
        {"reshaper_kwargs": {"add_kl": 1}},
        {"reshaper_kwargs": {"latent_dim_ratio": float("nan")}},
        {"reshaper_kwargs": {"latent_dim_ratio": float("inf")}},
        {"reshaper_kwargs": {"latent_dim_ratio": 0.0}},
        {"cls_token_regularizer_kwargs": {"unknown": 1}}, 
        {"cls_token_regularizer_kwargs": {
            "start": 0, "end": 1, "mlp_ratio": 0
        }},
    )
    for overrides in invalid_cases:
        try:
            DiffusionTransformer(build=False, **{**base, **overrides})
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Expected invalid configuration to fail: {overrides}")
    try:
        DiffusionTransformer(
            depth=1,
            vit_block_ids=[],
            reshaper_ids_dict={1: "flatten"},
            reshaper_kwargs={
                "add_kl": True,
                "latent_dim_ratio": 1e-12,
            },
            build=False,
            **base,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A zero-width transformer latent must fail.")
    for bad_resolution in (0, 3, 4.5):
        try:
            depth_zero.set_current_resolution(bad_resolution)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Expected invalid resolution: {bad_resolution}")
    try:
        depth_zero.encode(inputs, min_depth=1)
    except AssertionError:
        pass
    else:
        raise AssertionError("min_depth greater than depth must be rejected")
    try:
        progressive.add_depths("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown progressive layers must be rejected")

    tf.keras.backend.clear_session()
    return {"DiffusionTransformer": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
