"""Diffusion transformer with an attached feature-based classifier branch.

The class in this module remains a raw network: it predicts diffusion noise and
class probabilities.  Training objectives, schedules, EMA, and sampling live in
``diffusion.models.wrapper.diffusion_classifier``.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from copy import deepcopy

from typing import Literal

from . import CondType, TokenType, IdsType, IdsDictType

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer


def _select_first_token(x: tf.Tensor) -> tf.Tensor:
    """Select the class token from a token sequence.

    Args:
        x (tf.Tensor): Float token tensor of shape ``[B, tokens, features]``.

    Returns:
        tf.Tensor: First-token features of shape ``[B, features]``.
    """

    return x[:, 0, :]


class DiTClassifier(DiffusionTransformer):
    """Attach a configurable classifier pipeline to a diffusion transformer.

    The inherited ``layers_dicts`` form the noise-prediction branch.  The
    classifier obtains selected main-branch features through feature
    aggregators, processes them through ``clf_depth`` stages, passes the final
    selected classifier feature through a terminal connector at classifier
    depth ``clf_depth + 1``, pools a token/vector, and emits a softmax.

    Main-transformer depth 0 is the embedded input and 1..N are main stage
    outputs.  Classifier depth 0 is the first aggregated/main-derived input;
    classifier depths 1..``clf_depth`` are processing stages.  Consequently
    ``clf_layers_dicts`` has ``clf_depth + 1`` entries: the last entry is the
    terminal extraction connector, not another value counted by ``clf_depth``.
    ``feature_aggregation_ids_dict`` reads main depths, whereas
    ``clf_connection_ids_dict`` reads classifier depths.

    Every ``clf_`` option configures the classifier branch.  Values of ``None``
    inherit the corresponding noise-branch setting only for options handled by
    :meth:`_set_defaults`; branch-defining IDs, ``clf_dim``, condition/token
    modes, key/value widths, normalization MLP ratio, reshaper IDs, and
    regularizer IDs retain their explicit defaults.  See ``__init__`` for the
    exact initial state.

    Use :class:`diffusion.models.wrapper.diffusion_classifier.DiffusionClassifier`
    to optimize the noise and classifier losses jointly, or its V2 subclass to
    split variables between generator and discriminator optimizers.

    Attributes:
        clf_layers_dicts (list[dict[str, tf.keras.layers.Layer]]): Classifier
            processing dictionaries plus the terminal connector dictionary.
        classifier_feature_extractor (tf.keras.layers.Layer): First-token
            selector or global average pool.
        classifier (tf.keras.Sequential): Optional hidden layer/dropout and a
            final ``num_classes`` softmax.
        first_aggregated_dim (int): Width entering classifier depth 0.
        clf_grid_size (int): Spatial side inferred for classifier token layers.
        max_encoder_num (int): Greatest main feature depth needed to classify.
    """

    FA  = "0_feature_aggregator"
    FC  = "1_feature_connector"
    CAA = "2_cross_attention_aggregator"
    CAC = "3_cross_attention_connector"
    VTB = "4_vision_transformer_block"
    LM  = "5_local_mixer"
    DS  = "6_downsampler"
    US  = "7_upsampler"
    R   = "8_reshaper"
    CTR = "9_cls_token_regularizer"

    def __init__(
        self, 
        aggregate_from_noises: bool = False, 
        feature_aggregation_ids_dict: IdsDictType = {1: (-1,)}, 
        feature_aggregation_kwargs: dict = {}, 
        cross_attention_aggregation_ids_dict: IdsDictType = {}, 
        cross_attention_aggregation_kwargs: dict = {}, 
        classifier_only_cls_token: bool = True, 
        clf_dim: int | None = None, 
        clf_dim_forced: bool = False, 
        clf_cond_type: CondType | None = "time_label", 
        clf_cls_token_type: TokenType | None = "new_weight", 
        clf_depth: int = 1, 
        clf_connection_ids_dict: IdsDictType = {-1: (-1,)}, 
        clf_connection_kwargs: dict | None = None, 
        clf_cross_attention_ids_dict: IdsDictType = {}, 
        clf_cross_attention_kwargs: dict | None = None, 
        clf_cross_attention_plug_type: Literal["values", "queries"] | None = None, 
        clf_vit_block_ids: IdsType = [None], 
        clf_use_decoder_ids: IdsType = [], 
        clf_mha_key_dim: int | None = None, 
        clf_mha_value_dim: int | None = None, 
        clf_mha_num_heads: int | None = None, 
        clf_vit_block_mlp_ratio: float | None = None, 
        clf_vit_block_mlp_output_dims: dict[int, int] | None = None,
        clf_ln_mlp_ratio: float | None = None, 
        clf_ln_no_adaptation: bool | None = None, 
        clf_drop_prob: float | None = None, 
        clf_drop_per_sample: bool | None = None, 
        clf_local_mixer_ids: IdsType = [], 
        clf_local_mixer_kwargs: dict | None = None, 
        clf_downsample_ids: IdsType = [], 
        clf_downsample_kwargs: dict | None = None, 
        clf_upsample_ids: IdsType = [], 
        clf_upsample_kwargs: dict | None = None, 
        clf_reshaper_ids_dict: dict[int, str] = {},
        clf_reshaper_kwargs: dict = {}, 
        clf_cls_token_regularizer_ids: IdsType = [], 
        clf_cls_token_regularizer_kwargs: dict | None = None, 
        force_global_avg_pooling: bool = False, 
        classifier_mlp_ratio: int | None = None, 
        classifier_mlp_activation_func: str = "tanh", 
        dropout_rate: float = 0., 
        build: bool = True, 
        **kwargs: object
    ) -> None:
        """Initialize the joint noise-prediction and classifier network.

        Args:
            aggregate_from_noises (bool): Classify the network's predicted noise
                image rather than selected internal main-branch features.  This
                requires ``use_unpatchify=True``.
            feature_aggregation_ids_dict (dict[int, list[int | None]]): Maps a
                classifier target depth to main-transformer feature depths.
                Main depth 0 is embedded input; 1..N are stage outputs.  The
                default ``{1: (-1,)}`` resolves ``-1`` to the final main depth.
                Key 1 is mandatory.  ``None`` expands all main depths.
            feature_aggregation_kwargs (dict[str, object]): Shared aggregation
                options.  Accepted keys are ``connect_axis``, ``connect_type``,
                ``use_layer_norm``, ``ln_dim``, ``ln_mlp_ratio``,
                ``ln_no_adaptation``, ``mlp_output_dim``, ``mlp_ratio``, and
                ``mlp_activation_func``, with the same values and behavior as
                ``DiffusionTransformer.connection_kwargs``.
            cross_attention_aggregation_ids_dict (dict[int, list[int | None]]):
                Maps classifier target depths to main features used as external
                attention queries/values rather than the primary feature path.
            cross_attention_aggregation_kwargs (dict[str, object]): Same exact
                allowed keys as ``feature_aggregation_kwargs``.
            classifier_only_cls_token (bool): Give the classifier its own prefix
                token and suppress the main branch's token.  False lets both
                branches share the inherited main token behavior.
            clf_dim (int | None): Nominal classifier width.  ``None`` is replaced
                after aggregation by ``first_aggregated_dim``.
            clf_dim_forced (bool): Project feature merges/spatial width changes
                back to ``clf_dim``.  If true, ``clf_dim`` must be supplied.
            clf_cond_type (CondType | None): Classifier adaptive condition:
                ``"time_label"`` (default), ``"time"``, ``"label"``, or None.
                This value does not inherit ``cond_type``.
            clf_cls_token_type (TokenType | None): Classifier token source:
                ``"new_weight"`` (default), ``"time_label"``, ``"time"``,
                ``"label"``, or None.  It does not inherit ``cls_token_type``.
            clf_depth (int): Number of classifier processing depths; default 1.
                A terminal connector is created separately at depth
                ``clf_depth + 1``.
            clf_connection_ids_dict (dict[int, list[int | None]]): Classifier
                self-connections keyed by classifier target depth.  The special
                key ``-1`` is mandatory at construction and is moved to
                ``clf_depth + 1`` for the terminal connector.  Its default
                source ``(-1,)`` resolves to the last classifier processing
                depth.  Example ``{2: [0, 1], -1: [-1]}`` also merges depth 0
                and 1 before classifier stage 2.
            clf_connection_kwargs (dict[str, object] | None): Connector keys
                listed for ``feature_aggregation_kwargs``.  ``None`` inherits
                the main branch's ``connection_kwargs``; ``{}`` requests layer
                defaults/inferred dimensions.
            clf_cross_attention_ids_dict (dict[int, list[int | None]]): Maps a
                classifier stage to earlier classifier features used for cross
                attention.  Empty by default.
            clf_cross_attention_kwargs (dict[str, object] | None): Cross-
                attention connector options; ``None`` inherits the main branch.
            clf_cross_attention_plug_type (Literal["values", "queries"] | None):
                External-attention plug side.  ``None`` inherits the main
                ``cross_attention_plug_type`` (default ``"values"``).
            clf_vit_block_ids (list[int | None]): Classifier attention-block
                depths.  ``[None]`` means all 1..``clf_depth``; ``[]`` means no
                blocks.  This list does not inherit from the main branch.
            clf_use_decoder_ids (list[int | None]): Classifier block depths that
                use ``DiTDecoderBlock``.  Empty by default.
            clf_mha_key_dim (int | None): Classifier per-head key width.  It
                remains ``None`` by default and is inferred by the block.
            clf_mha_value_dim (int | None): Classifier per-head value width;
                remains ``None`` by default.
            clf_mha_num_heads (int | None): Head count; ``None`` inherits
                ``mha_num_heads`` (4 by default).
            clf_vit_block_mlp_ratio (float | None): Classifier FFN expansion;
                ``None`` inherits ``vit_block_mlp_ratio`` (4 by default).
            clf_vit_block_mlp_output_dims (dict[int, int] | None): Optional
                classifier per-depth output widths; ``None`` copies the main
                mapping, while ``{}`` explicitly requests none.
            clf_ln_mlp_ratio (float | None): Classifier adaptive-normalization
                MLP ratio.  This explicit default remains None; it does not
                inherit ``ln_mlp_ratio``.
            clf_ln_no_adaptation (bool | None): Disable condition adaptation;
                ``None`` inherits the main setting (false by default).
            clf_drop_prob (float | None): Residual-drop probability; ``None``
                inherits the main value (0 by default).
            clf_drop_per_sample (bool | None): Drop residuals per sample;
                ``None`` inherits the main value (true by default).
            clf_local_mixer_ids (list[int | None]): Classifier local-mixer
                depths; empty by default and independent of main IDs.
            clf_local_mixer_kwargs (dict[str, object] | None): Exact keys from
                ``local_mixer_kwargs``; ``None`` inherits the main mapping.
            clf_downsample_ids (list[int | None]): Classifier downsample depths.
            clf_downsample_kwargs (dict[str, object] | None): Exact keys from
                ``downsample_kwargs``; ``None`` inherits the main mapping.
            clf_upsample_ids (list[int | None]): Classifier upsample depths.
            clf_upsample_kwargs (dict[str, object] | None): Exact keys from
                ``upsample_kwargs``; ``None`` inherits the main mapping.
            clf_reshaper_ids_dict (dict[int, str]): Classifier depth to
                ``"flatten"``/``"unflatten"`` mapping; empty by default.
            clf_reshaper_kwargs (dict[str, object]): ``add_kl`` (bool) and
                ``latent_dim_ratio`` (positive float).  The default empty
                mapping is classifier-specific and does not inherit main values.
            clf_cls_token_regularizer_ids (list[int | None]): Classifier depths
                0..``clf_depth`` with auxiliary class softmax heads.  Empty by
                default; ``[None]`` selects the full range.
            clf_cls_token_regularizer_kwargs (dict[str, int] | None): ``start``
                and ``end`` token-slice bounds.  ``None`` inherits the main
                mapping, normally ``{"start": 0, "end": 1}``.
            force_global_avg_pooling (bool): Average all final tokens even when
                a class token is available.  Without a usable class token,
                global average pooling is selected automatically.
            classifier_mlp_ratio (int | None): Add a hidden classifier Dense
                layer of ``final_width * ratio`` units when non-None.
            classifier_mlp_activation_func (str | callable): Hidden classifier
                activation, default ``"tanh"``.
            dropout_rate (float): Classifier dropout rate; 0 omits dropout.
            build (bool): Build symbolic inputs and variables immediately.
            **kwargs (object): All ``DiffusionTransformer`` constructor options plus
                standard Keras model options.  Main-branch ``cls_token_type``
                is intercepted according to ``classifier_only_cls_token``.

        Returns:
            None: Both branches and the classifier head are initialized.
        """

        temp_val = kwargs.pop("cls_token_type", None)
        super().__init__(
            cls_token_type=None if classifier_only_cls_token and \
                        temp_val is not None else temp_val, 
            build=False, 
            **kwargs
        )
        self._check_clf_assertions(locals())
        self._save_init_args(locals())
        self._set_defaults(locals())
        self._handle_all_clf_ids()
        self.set_max_encoder_num()

        self.first_aggregated_dim = self._get_unforced_total_dim(
            ids_set=self.feature_aggregation_ids_dict[1], 
            layers_dicts=self.layers_dicts, 
            base_dim=self.dim, 
            kwargs=self.feature_aggregation_kwargs
        ) if not self.clf_dim_forced and 1 not in self.clf_connection_ids_dict else self.clf_dim
        self.first_aggregated_dim = (
            self.patches_dim if self.classifier_only_cls_token else self.dim
        ) if self.aggregate_from_noises else self.first_aggregated_dim
        self.clf_dim = self.first_aggregated_dim if self.clf_dim is None else self.clf_dim
        self.clf_grid_size = self._get_ids_grid_size(
            ids_set=self.feature_aggregation_ids_dict[1], 
            layers_dicts=self.layers_dicts, 
            base_grid_size=self.grid_size, 
            must_be_same=True
        ) if not self.aggregate_from_noises else self.grid_size
        self.clf_connection_ids_dict[self.clf_depth+1] = self.clf_connection_ids_dict.pop(-1, (-1,))

        self._create_clf_embedders()
        self.cls_token = self._create_cls_token(
            self.first_aggregated_dim if self.aggregate_from_noises else self.clf_dim, 
            self.cls_token_pos_merger_type, 
            self.cls_token_freq_dim, 
            self.cls_token_mlp_ratio, 
            self.clf_cls_token_type, 
            "clf_"
        ) if self.classifier_only_cls_token else self.cls_token
        self._create_clf_layers()

        # Pool all tokens when requested or when no usable class token exists.
        if self.force_global_avg_pooling or not (
        (not self.classifier_only_cls_token and \
        self.cls_token_type is not None) or \
        (self.classifier_only_cls_token and \
        self.clf_cls_token_type is not None)):
            self.classifier_feature_extractor = layers.GlobalAveragePooling1D(
                name=f"{self.name_prefix}classifier_feature_extractor"
            )
        # Otherwise classify from the first class-token position.
        else:
            self.classifier_feature_extractor = layers.Lambda(
                _select_first_token,
                name=f"{self.name_prefix}classifier_feature_extractor"
            )

        self.classifier = models.Sequential( # TODO: build it as a functional model
            name=f"{self.name_prefix}classes"
        )
        # Add the optional hidden classifier projection.
        if self.classifier_mlp_ratio is not None:
            self.classifier.add(layers.Dense(
                self._get_unforced_total_dim(
                    ids_set=[self.clf_depth], 
                    layers_dicts=self.clf_layers_dicts, 
                    base_dim=self.clf_dim, 
                ) * self.classifier_mlp_ratio, 
                activation=self.classifier_mlp_activation_func, 
                name=f"{self.classifier.name}/first_layer"
            ))
        # Add classifier dropout only for a nonzero rate.
        if self.dropout_rate > 0.:
            self.classifier.add(layers.Dropout(
                self.dropout_rate, 
                name="dropout_layer"
            ))
        self.classifier.add(layers.Dense(
            self.num_classes, 
            activation="softmax", 
            name=f"{self.classifier.name}/final_layer"
        ))

        # Materialize classifier and denoiser variables when requested.
        if self.build_:
            self.build()

    def _check_clf_assertions(self, local_vars: dict) -> None:
        """Validate classifier-specific branch configuration.

        Args:
            local_vars (dict[str, object]): ``__init__`` locals after the main
                transformer has been initialized.

        Returns:
            None: Invalid CFG requirements, missing mandatory aggregator or
            terminal IDs, out-of-range depths, unsupported kwargs keys, and
            invalid attention plug types raise ``AssertionError``.
        """

        local_vars["depth"] = self.depth

        assert self.use_cfg, \
            "use_cfg must be True for classification to work."

        # Noise aggregation requires image-shaped denoiser output.
        if local_vars["aggregate_from_noises"]:
            assert self.use_unpatchify, \
                "aggregate_from_noises requires use_unpatchify to be True."

        assert 1 in local_vars["feature_aggregation_ids_dict"] and \
            "There must be at least one feature vector to connect to the classifier part."
        self._check_dict_assertions(
            local_vars, 
            "feature_aggregation_ids_dict", 
            depth_name="clf_depth", 
            id_less_than_key=False
        )
        self._check_dict_assertions(
            local_vars, 
            "feature_aggregation_kwargs", 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=self.feature_handler_kwargs_allowed_vals, 
            check_values=False, 
        )
        self._check_dict_assertions(
            local_vars, 
            "cross_attention_aggregation_ids_dict", 
            depth_name="clf_depth", 
            id_less_than_key=False
        )
        self._check_dict_assertions(
            local_vars, 
            "cross_attention_aggregation_kwargs", 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=self.feature_handler_kwargs_allowed_vals, 
            check_values=False, 
        )

        assert -1 in local_vars["clf_connection_ids_dict"], \
            "There must be at least one feature vector to extract from the classifier part."
        self._check_dict_assertions(
            local_vars, 
            "clf_connection_ids_dict", 
            depth_name="clf_depth", 
            second_depth_name="clf_depth", 
            check_items_num=False, 
            check_keys=False, 
            id_less_than_key=False
        )
        self._check_dict_assertions(
            local_vars, 
            key, 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=self.feature_handler_kwargs_allowed_vals, 
            check_values=False, 
        ) if local_vars[key:="clf_connection_kwargs"] is not None else None
        self._check_dict_assertions(
            local_vars, 
            "clf_cross_attention_ids_dict", 
            depth_name="clf_depth", 
            second_depth_name="clf_depth"
        )
        self._check_dict_assertions(
            local_vars, 
            key, 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=self.feature_handler_kwargs_allowed_vals, 
            check_values=False, 
        ) if local_vars[key:="clf_cross_attention_kwargs"] is not None else None
        self._check_dict_assertions(
            local_vars, 
            "clf_vit_block_ids", 
            id_less_than_key=False, 
            depth_name="clf_depth", 
            second_depth_name="clf_depth"
        )
        self._check_dict_assertions(
            local_vars, 
            "clf_use_decoder_ids", 
            id_less_than_key=False, 
            depth_name="clf_depth", 
            second_depth_name="clf_depth"
        )
        self._check_dict_assertions(
            local_vars, 
            key, 
            check_values=False, 
            depth_name="clf_depth", 
            second_depth_name="clf_depth"
        ) if local_vars[key:="clf_vit_block_mlp_output_dims"] is not None else None
        self._check_dict_assertions(
            local_vars, 
            "clf_local_mixer_ids", 
            id_less_than_key=False, 
            depth_name="clf_depth", 
            second_depth_name="clf_depth"
        )
        self._check_dict_assertions(
            local_vars, 
            key, 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=self.local_mixer_kwargs_allowed_vals, 
            check_values=False, 
        ) if local_vars[key:="clf_local_mixer_kwargs"] is not None else None
        self._check_dict_assertions(
            local_vars, 
            "clf_downsample_ids", 
            id_less_than_key=False, 
            depth_name="clf_depth", 
            second_depth_name="clf_depth"
        )
        self._check_dict_assertions(
            local_vars, 
            key, 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=self.downsample_kwargs_allowed_vals, 
            check_values=False, 
        ) if local_vars[key:="clf_downsample_kwargs"] is not None else None
        self._check_dict_assertions(
            local_vars, 
            "clf_upsample_ids", 
            id_less_than_key=False, 
            depth_name="clf_depth", 
            second_depth_name="clf_depth"
        )
        self._check_dict_assertions(
            local_vars, 
            key, 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=self.upsample_kwargs_allowed_vals, 
            check_values=False, 
        ) if local_vars[key:="clf_upsample_kwargs"] is not None else None
        self._check_dict_assertions(
            local_vars, 
            "clf_reshaper_ids_dict", 
            id_less_than_key=False, 
            check_values=False, 
            none_is_filler=False, 
            depth_name="clf_depth", 
            second_depth_name="clf_depth", 
        )
        self._check_dict_assertions(
            local_vars, 
            key, 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=self.reshaper_kwargs_allowed_vals, 
            check_values=False, 
        ) if local_vars[key:="clf_reshaper_kwargs"] is not None else None
        self._check_dict_assertions(
            local_vars, 
            "clf_cls_token_regularizer_ids", 
            id_less_than_key=False, 
            depth_name="clf_depth", 
            second_depth_name="clf_depth", 
            allowed_values=[None]+list(range(local_vars["clf_depth"]+1))
        )
        self._check_dict_assertions(
            local_vars, 
            key, 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=self.cls_token_regularizer_kwargs_allowed_vals, 
            check_values=False, 
        ) if local_vars[key:="clf_cls_token_regularizer_kwargs"] is not None else None

        assert local_vars["clf_cross_attention_plug_type"] \
            in (None, "values", "queries"), \
            "clf_cross_attention_plug_type can only be values or queries."

    def _set_defaults(self, local_vars: dict, 
                    exclude: list[str]=["clf_dim", "clf_cond_type", 
                        "clf_connection_ids_dict", "clf_cross_attention_ids_dict", 
                        "clf_vit_block_ids", "clf_use_decoder_ids", 
                        "clf_local_mixer_ids", "clf_downsample_ids", 
                        "clf_upsample_ids", "clf_cls_token_type", 
                        "clf_mha_key_dim", "clf_mha_value_dim", 
                        "clf_ln_mlp_ratio", "clf_reshaper_ids_dict", 
                        "clf_cls_token_regularizer_ids"]) -> None:
        """Resolve inheritable ``clf_*`` values from main-branch attributes.

        Args:
            local_vars (dict[str, object]): Constructor namespace containing
                classifier values before resolution.
            exclude (list[str]): Classifier option names that must retain their
                explicit values, including branch-defining IDs, widths, token
                modes, key/value dimensions, and reshaper/regularizer IDs.

        Returns:
            None: For each non-excluded ``clf_name``, a None value becomes the
            current ``self.name`` value; non-None values are retained.

        Raises:
            AssertionError: If ``clf_dim_forced=True`` but ``clf_dim`` is None.
        """

        for name, clf_part_value in local_vars.items():
            # Validate only classifier-specific constructor fields not explicitly excluded.
            if not name.startswith("clf_") or name in exclude:
                continue

            noise_part_name = name.replace("clf_", '')
            noise_part_value = getattr(self, noise_part_name)
            clf_part_value = noise_part_value if clf_part_value is None else clf_part_value

            setattr(self, name, clf_part_value)

        # A forced classifier width requires an explicit target width.
        if self.clf_dim_forced:
            assert self.clf_dim is not None, \
                "When clf_dim_forced is true, clf_dim cannot be None."

    def _handle_all_clf_ids(self) -> None:
        """Normalize every classifier and main-feature aggregation ID set.

        Classifier component selections expand over depths 1..``clf_depth``;
        classifier regularizer ``None`` expands over 0..``clf_depth``.  Main
        aggregators resolve negative IDs against main ``depth`` and classifier
        connectors resolve them against ``clf_depth``.

        Returns:
            None: ID attributes are normalized in place.
        """

        self.clf_vit_block_ids = self._handle_ids(
            self.clf_vit_block_ids, 
            depth=self.clf_depth, 
            min_id=1, 
            max_id=self.clf_depth
        )
        self.clf_use_decoder_ids = self._handle_ids(
            self.clf_use_decoder_ids, 
            depth=self.clf_depth, 
            min_id=1, 
            max_id=self.clf_depth
        )
        self.clf_local_mixer_ids = self._handle_ids(
            self.clf_local_mixer_ids, 
            depth=self.clf_depth, 
            min_id=1, 
            max_id=self.clf_depth
        )
        self.clf_downsample_ids = self._handle_ids(
            self.clf_downsample_ids, 
            depth=self.clf_depth, 
            min_id=1, 
            max_id=self.clf_depth
        )
        self.clf_upsample_ids = self._handle_ids(
            self.clf_upsample_ids, 
            depth=self.clf_depth, 
            min_id=1, 
            max_id=self.clf_depth
        )
        self.clf_cls_token_regularizer_ids = self._handle_ids(
            self.clf_cls_token_regularizer_ids, 
            depth=self.clf_depth, 
            min_id=0, 
            max_id=self.clf_depth
        )
        self._handle_ids(
            self.feature_aggregation_ids_dict, 
            depth=self.depth, 
            max_id=self.depth
        )
        self._handle_ids(
            self.cross_attention_aggregation_ids_dict, 
            depth=self.depth, 
            max_id=self.depth
        )
        self._handle_ids(
            self.clf_connection_ids_dict, 
            depth=self.clf_depth, 
            max_id=None
        )
        self._handle_ids(
            self.clf_cross_attention_ids_dict, 
            depth=self.clf_depth, 
            max_id=None
        )

    def _get_layers_dict_last_output_dim(self, layers_dict: dict, 
                                        skip_reshaper: bool) -> int | None:
        """Return the final feature width from a classifier/main stage mapping.

        Args:
            layers_dict (dict[str, tf.keras.layers.Layer]): Stage components.
            skip_reshaper (bool): Ignore reshaper output widths.

        Returns:
            int | None: Feature-aggregator width when it is the latest relevant
            component, otherwise the width resolved by the base implementation.
        """

        last_output_dim = None

        # Account for the classifier feature aggregator's output width.
        if (key:=self.FA) in layers_dict:
            last_output_dim = layers_dict[key].output_dim

        last_output_dim = last_output_dim if (last_output_dim_:=
            super()._get_layers_dict_last_output_dim(
                layers_dict, 
                skip_reshaper
        )) is None else last_output_dim_

        return last_output_dim

    def _create_clf_embedders(self) -> None:
        """Create or reuse condition layers needed by the classifier branch.

        With ``classifier_only_cls_token=True``, unused main token-conditioning
        embedders may be removed and ``cls_token_type`` is cleared.  Existing
        time/label embedders are shared when compatible; otherwise classifier-
        named embedders are created.  A classifier depth-0 regularizer is also
        created when requested.

        Returns:
            None: Embedder, merger, and ``clf_labels_embed_reg`` attributes are
            assigned in place.
        """

        self._clf_cond_type = self.clf_cond_type if self.clf_cond_type is not None and not self.clf_ln_no_adaptation else []
        self._clf_cls_token_type = self.clf_cls_token_type if self.clf_cls_token_type is not None else []

        clf_embed_times_flag = "time" in self._clf_cls_token_type or "time" in self._clf_cond_type
        clf_embed_labels_flag = "label" in self._clf_cls_token_type or "label" in self._clf_cond_type
        clf_conds_merger_flag = ("time" in self._clf_cls_token_type and "label" in self._clf_cls_token_type
                                ) or ("time" in self._clf_cond_type and "label" in self._clf_cond_type)

        # Remove main-branch token dependencies used exclusively by the classifier.
        if self.classifier_only_cls_token:
            # Drop an otherwise unused main time embedder.
            if flag1:=("time" in self._cls_token_type) and not clf_embed_times_flag:
                self.time_embedder = None
            # Drop an otherwise unused main label embedder.
            if flag2:=("label" in self._cls_token_type) and not clf_embed_labels_flag:
                self.label_embedder = None
            # Drop the main condition merger when both component embedders were removed.
            if flag1 and flag2 and not clf_conds_merger_flag:
                self.conds_merger = None
            self.cls_token_type = None
            self._cls_token_type = []

        self.time_embedder = self._create_time_embedder(
            name_prefix="clf_"
        ) if clf_embed_times_flag and self.time_embedder is None else self.time_embedder

        self.label_embedder = self._create_label_embedder(
            name_prefix="clf_"
        ) if clf_embed_labels_flag and self.label_embedder is None else self.label_embedder

        self.conds_merger = self._create_merger(
            merger_type=self.conds_merger_type, 
            name=f"{self.name_prefix}clf_depth_0_time_label_merger"
        ) if clf_conds_merger_flag and self.conds_merger is None else self.conds_merger

        self.clf_labels_embed_reg = self._create_token_regularizer(
            name=f"{self.name_prefix}clf_depth_0_{self.CTR[2:]}"
        ) if 0 in self.clf_cls_token_regularizer_ids else None

    def _create_clf_layer_dict(
        self, 
        i: int, 
        layers_dicts: list[dict]
    ) -> dict[str, layers.Layer]:
        """Construct one classifier stage or its terminal connector.

        Args:
            i (int): Zero-based classifier stage index.  Public key ``i+1`` may
                be 1..``clf_depth`` or terminal key ``clf_depth+1``.
            layers_dicts (list[dict]): Previously built classifier stages.

        Returns:
            dict[str, tf.keras.layers.Layer | tf.keras.Model]: Components in
            aggregator, classifier connector, cross-attention aggregation,
            cross-attention connector, transformer, mixer, scaler, reshaper,
            regularizer order.  Aggregators read ``self.layers_dicts`` (main
            features); ``clf_`` connectors read classifier features.
        """
        layers_dict = {}
        key = i+1

        # Build the classifier's main-feature aggregator, bypassing it for noise input.
        if key in self.feature_aggregation_ids_dict:
            bypass_aggregator = self.aggregate_from_noises and key == 1
            layers_dict[self.FA] = self._create_feature_handler(
                ids_set=self.feature_aggregation_ids_dict[key] 
                        if not bypass_aggregator
                        else [], 
                layers_dicts=self.layers_dicts, 
                base_dim=self.clf_dim, 
                dim_forced=False if bypass_aggregator else self.clf_dim_forced, 
                ln_mlp_ratio=self.clf_ln_mlp_ratio, 
                ln_no_adaptation=self.clf_ln_no_adaptation, 
                zero_index_base_dim=self.dim, 
                increased_dim=self._get_last_output_dim(
                    i=i-1, 
                    layers_dicts=layers_dicts, 
                    base_dim=self.clf_dim
                ) if key not in self.clf_connection_ids_dict and i != 0 \
                  else self.first_aggregated_dim if bypass_aggregator else 0, 
                output_dim_flag=key not in self.clf_connection_ids_dict, 
                kwargs={} if bypass_aggregator else self.feature_aggregation_kwargs, 
                name=f"{self.name_prefix}clf_depth_{key}_{self.FA[2:]}"
            )

        # Build this classifier depth's residual feature connector.
        if key in self.clf_connection_ids_dict:
            layers_dict[self.FC] = self._create_feature_handler(
                ids_set=self.clf_connection_ids_dict[key], 
                layers_dicts=layers_dicts, 
                base_dim=self.clf_dim, 
                dim_forced=self.clf_dim_forced, 
                ln_mlp_ratio=self.clf_ln_mlp_ratio, 
                ln_no_adaptation=self.clf_ln_no_adaptation, 
                zero_index_base_dim=self.first_aggregated_dim, 
                increased_dim=self._get_unforced_total_dim(
                    ids_set=self.feature_aggregation_ids_dict.get(key, []), 
                    layers_dicts=self.layers_dicts, 
                    base_dim=self.dim, 
                    kwargs=self.feature_aggregation_kwargs
                ) if not (self.aggregate_from_noises and key == 1) 
                else self.first_aggregated_dim, 
                kwargs=self.clf_connection_kwargs, 
                name=f"{self.name_prefix}clf_depth_{key}_{self.FC[2:]}"
            )

        # Build the classifier's cross-attention feature aggregator.
        if key in self.cross_attention_aggregation_ids_dict:
            layers_dict[self.CAA] = self._create_feature_handler(
                ids_set=self.cross_attention_aggregation_ids_dict[key], 
                layers_dicts=self.layers_dicts, 
                base_dim=self.clf_dim, 
                dim_forced=self.clf_dim_forced, 
                ln_mlp_ratio=self.clf_ln_mlp_ratio, 
                ln_no_adaptation=self.clf_ln_no_adaptation, 
                zero_index_base_dim=self.dim, 
                output_dim_flag=key not in self.clf_cross_attention_ids_dict, 
                kwargs=self.cross_attention_aggregation_kwargs, 
                name=f"{self.name_prefix}clf_depth_{key}_{self.CAA[2:]}"
            )

        # Build this classifier depth's residual cross-attention connector.
        if key in self.clf_cross_attention_ids_dict:
            layers_dict[self.CAC] = self._create_feature_handler(
                ids_set=self.clf_cross_attention_ids_dict[key], 
                layers_dicts=layers_dicts, 
                base_dim=self.clf_dim, 
                dim_forced=self.clf_dim_forced, 
                ln_mlp_ratio=self.clf_ln_mlp_ratio, 
                ln_no_adaptation=self.clf_ln_no_adaptation, 
                zero_index_base_dim=self.first_aggregated_dim, 
                increased_dim=self._get_unforced_total_dim(
                    ids_set=self.cross_attention_aggregation_ids_dict.get(key, []), 
                    layers_dicts=self.layers_dicts, 
                    base_dim=self.dim, 
                    kwargs=self.cross_attention_aggregation_kwargs
                ), 
                kwargs=self.clf_cross_attention_kwargs, 
                name=f"{self.name_prefix}clf_depth_{key}_{self.CAC[2:]}"
            )

        # Build a classifier transformer block with the preceding handler width.
        if key in self.clf_vit_block_ids:
            mha_query_dim = layers_dict[self.CAA].output_dim if \
                            key in self.cross_attention_aggregation_ids_dict and \
                            self.clf_cross_attention_plug_type == "queries" \
                            else None
            mha_query_dim = layers_dict[self.CAC].output_dim if \
                            key in self.clf_cross_attention_ids_dict and \
                            self.clf_cross_attention_plug_type == "queries" \
                            else mha_query_dim

            layers_dict[self.VTB] = self._create_vit_block(
                i=i, layers_dicts=layers_dicts, 
                layers_dict=layers_dict, base_dim=self.clf_dim, 
                mha_key_dim=self.clf_mha_key_dim, 
                mha_value_dim=self.clf_mha_value_dim, 
                mha_query_dim=mha_query_dim, 
                mha_num_heads=self.clf_mha_num_heads, 
                mlp_ratio=self.clf_vit_block_mlp_ratio, 
                mlp_output_dim=self.clf_vit_block_mlp_output_dims.get(key, None), 
                ln_mlp_ratio=self.clf_ln_mlp_ratio, 
                ln_no_adaptation=self.clf_ln_no_adaptation, 
                drop_prob=self.clf_drop_prob, 
                drop_per_sample=self.clf_drop_per_sample, 
                use_decoder=key in self.clf_use_decoder_ids, 
                name_prefix=f"{self.name_prefix}clf_depth_{key}_"
            )

        # Build this classifier depth's local mixer.
        if key in self.clf_local_mixer_ids:
            layers_dict[self.LM] = self._create_local_mixer(
                i=i, 
                dim_forced=self.clf_dim_forced, 
                layers_dicts=layers_dicts, 
                layers_dict=layers_dict, 
                base_dim=self.clf_dim, 
                base_grid_size=self.clf_grid_size, 
                ln_mlp_ratio=self.clf_ln_mlp_ratio, 
                ln_no_adaptation=self.clf_ln_no_adaptation, 
                circumvent_cls_token=(self.classifier_only_cls_token and \
                                    self.clf_cls_token_type is not None) or \
                                    (not self.classifier_only_cls_token and \
                                    self.cls_token_type is not None),
                kwargs=self.clf_local_mixer_kwargs, 
                name=f"{self.name_prefix}clf_depth_{key}_{self.LM[2:]}"
            )

        # Build this classifier depth's downsampler.
        if key in self.clf_downsample_ids:
            layers_dict[self.DS] = self._create_scaler(
                scaler_type="downsample", 
                i=i, 
                dim_forced=self.clf_dim_forced, 
                layers_dicts=layers_dicts, 
                layers_dict=layers_dict, 
                base_dim=self.clf_dim, 
                base_grid_size=self.clf_grid_size, 
                ln_mlp_ratio=self.clf_ln_mlp_ratio, 
                ln_no_adaptation=self.clf_ln_no_adaptation, 
                circumvent_cls_token=(self.classifier_only_cls_token and \
                                    self.clf_cls_token_type is not None) or \
                                    (not self.classifier_only_cls_token and \
                                    self.cls_token_type is not None),
                kwargs=self.clf_downsample_kwargs, 
                name=f"{self.name_prefix}clf_depth_{key}_{self.DS[2:]}"
            )

        # Build this classifier depth's upsampler.
        if key in self.clf_upsample_ids:
            layers_dict[self.US] = self._create_scaler(
                scaler_type="upsample", 
                i=i, 
                dim_forced=self.clf_dim_forced, 
                layers_dicts=layers_dicts, 
                layers_dict=layers_dict, 
                base_dim=self.clf_dim, 
                base_grid_size=self.clf_grid_size, 
                ln_mlp_ratio=self.clf_ln_mlp_ratio, 
                ln_no_adaptation=self.clf_ln_no_adaptation, 
                circumvent_cls_token=(self.classifier_only_cls_token and \
                                    self.clf_cls_token_type is not None) or \
                                    (not self.classifier_only_cls_token and \
                                    self.cls_token_type is not None),
                kwargs=self.clf_upsample_kwargs, 
                name=f"{self.name_prefix}clf_depth_{key}_{self.US[2:]}"
            )

        # Build this classifier depth's flatten or unflatten reshaper.
        if key in self.clf_reshaper_ids_dict:
            layers_dict[self.R] = self._create_reshaper(
                reshape_type=self.clf_reshaper_ids_dict[key], 
                i=i, layers_dicts=layers_dicts, 
                layers_dict=layers_dict, base_dim=self.clf_dim, 
                base_grid_size=self.clf_grid_size, 
                grid_has_cls_token=(self.clf_cls_token_type is not None and \
                                    self.classifier_only_cls_token) or \
                                    (self.cls_token_type is not None and \
                                    not self.classifier_only_cls_token), 
                kwargs=self.clf_reshaper_kwargs, 
                name=f"{self.name_prefix}clf_depth_{key}_{self.R[2:]}"
            )

        # Build this classifier depth's auxiliary token head.
        if key in self.clf_cls_token_regularizer_ids:
            layers_dict[self.CTR] = self._create_token_regularizer(
                name=f"{self.name_prefix}clf_depth_{key}_{self.CTR[2:]}"
            )

        return layers_dict

    def _create_clf_layers(self) -> None:
        """Create classifier processing stages and the terminal extraction stage.

        Returns:
            None: ``clf_layers_dicts`` receives ``clf_depth + 1`` dictionaries;
            the final dictionary normally contains the mandatory connector
            moved from constructor key ``-1``.
        """

        self.clf_layers_dicts = []

        for i in range(self.clf_depth+1):
            self.clf_layers_dicts.append(
                self._create_clf_layer_dict(i, self.clf_layers_dicts)
            )

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        full_return: bool = False, 
        training: bool | None = None
    ) -> dict:
        """Predict diffusion noise and class probabilities in one pass.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Noisy images
                ``[B,H,W,C]``, timestep IDs ``[B]``, and CFG label IDs ``[B]``.
            full_return (bool): Include both branches' intermediate tensors.
            training (bool | None): Keras training mode.

        Returns:
            dict[str, object]: Always ``{"noises": [B,H,W,C], "classes":
            [B,num_classes]}``.  With ``full_return=True``, also contains
            ``cond``, ``features_list``, ``regs_list``, ``z_vals`` for the main
            branch and ``clf_cond``, ``clf_features_list``, ``clf_regs_list``,
            ``clf_z_vals`` for the classifier branch.
        """

        noises, cond, features_list, regs_list, z_vals = super().call(
            inputs, 
            full_return=True, 
            training=training
        )
        outputs = self.compute_class(
            features_list, 
            noises, 
            times=inputs[1], 
            labels=inputs[2], 
            training=training
        )
        output_dict = {
            "noises": noises, 
        }

        # Include intermediate classifier metadata only for full returns.
        if full_return:
            output_dict["cond"] = cond
            output_dict["features_list"] = features_list
            output_dict["regs_list"] = regs_list
            output_dict["z_vals"] = z_vals
            output_dict["classes"] = outputs[0]
            output_dict["clf_cond"] = outputs[1]
            output_dict["clf_features_list"] = outputs[2]
            output_dict["clf_regs_list"] = outputs[3]
            output_dict["clf_z_vals"] = outputs[4]
        # Otherwise expose the classifier probabilities as the primary result.
        else:
            output_dict["classes"] = outputs[0]

        return output_dict

    def set_max_encoder_num(self, max_encoder_num: int | None = None) -> None:
        """Set how many main transformer stages classification must execute.

        Args:
            max_encoder_num (int | None): Explicit exclusive loop stop passed to
                :meth:`encode`.  ``None`` computes the greatest main depth used
                by either aggregation mapping, also considering the final depth
                when ``aggregate_from_noises=True``.

        Returns:
            None: ``self.max_encoder_num`` is assigned.
        """

        aggregation_ids = []
        [aggregation_ids.extend(value) for value in \
            self.feature_aggregation_ids_dict.values()]
        [aggregation_ids.extend(value) for value in \
            self.cross_attention_aggregation_ids_dict.values()]

        # Include the denoiser output as a classifier source when configured.
        if self.aggregate_from_noises:
            aggregation_ids.append(self.depth)

        self.max_encoder_num = max(
            aggregation_ids
        ) if max_encoder_num is None else max_encoder_num

    def predict_noise(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        full_return: bool = False, 
        training: bool | None = None
    ) -> tf.Tensor | tuple[
        tf.Tensor, tf.Tensor, list[tf.Tensor], list[tf.Tensor],
        tuple[tf.Tensor, tf.Tensor]
    ]:
        """Run only the inherited diffusion noise-prediction branch.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Images/noisy images
                ``[B,H,W,C]``, integer timesteps ``[B]``, labels ``[B]``.
            full_return (bool): Return base-branch intermediates when true.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor | tuple: Same output contract as
            ``DiffusionTransformer.call``; no classifier output is computed.
        """

        outputs = super().call(
            inputs, 
            full_return=full_return, 
            training=training
        )

        return outputs

    def compute_class(
        self, 
        features_list: list[tf.Tensor], 
        noises: tf.Tensor | None, 
        times: tf.Tensor, 
        labels: tf.Tensor, 
        training: bool | None = None
    ) -> tuple[tf.Tensor, tf.Tensor, list[tf.Tensor], 
        list[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
        """Compute class probabilities from main features or predicted noises.

        Args:
            features_list (list[tf.Tensor]): Main features indexed by absolute
                depth, with rank-3 token tensors at routed depths.
            noises (tf.Tensor | None): Predicted image/noise ``[B,H,W,C]``.
                Required when ``aggregate_from_noises=True``; otherwise ignored.
            times (tf.Tensor): Integer timestep IDs ``[B]``.
            labels (tf.Tensor): Integer condition label IDs ``[B]``.
            training (bool | None): Keras training mode.

        Returns:
            tuple: ``(classes, clf_cond, clf_features_list, clf_regs_list,
            clf_z_vals)``.  ``classes`` is float ``[B,num_classes]`` softmax;
            features index 0 is classifier depth 0, regularizer entries may be
            None, and latent statistics are ``(mean, log_variance)`` or
            ``(None, None)``.
        """

        clf_cond, time_embeds, label_embeds = self.embed_conditions(
            times, labels, 
            self.clf_cond_type, 
            full_return=True, 
            training=training
        )

        z = self.clf_labels_embed_reg(
            label_embeds, 
            training=training
        ) if self.clf_labels_embed_reg is not None else None

        clf_features_list = []
        clf_regs_list = [z]
        clf_z_vals = (None, None)
        for i, layers_dict in enumerate(self.clf_layers_dicts):
            x = layers_dict[self.FA](
                features_list, 
                [x] if self.FC not in layers_dict and i != 0 else [], 
                cond=clf_cond, 
                training=training
            ) if self.FA in layers_dict else x
        # Initialize the classifier branch at its first depth.
            if i == 0:
            # Start from the denoiser's predicted image/noise when configured.
                if self.aggregate_from_noises:
                # Embed noise patches without main conditions for a classifier-only token.
                    if self.classifier_only_cls_token:
                        x = self.patch_embedder(
                            noises, 
                            output_grid_size=self._current_resolution // self.patch_size if \
                                            self.image_size != self._current_resolution \
                                            else None, 
                            training=training
                        )
                # Otherwise embed noise with the shared main condition components.
                    else:
                        x, (_, main_time_embeds, main_label_embeds) = self.embed_inputs(
                            (noises, times, labels), 
                            self.cond_type, 
                            full_return=True, 
                            training=training
                        )
                        x = self.prepend_cls_token(
                            x, self.cls_token_type, 
                            time_embeds=main_time_embeds, 
                            label_embeds=main_label_embeds, 
                            times=times, labels=labels, 
                            training=training
                        ) if self.cls_token_type is not None else x

                x = self.prepend_cls_token(
                    x, self.clf_cls_token_type, 
                    time_embeds=time_embeds, 
                    label_embeds=label_embeds, 
                    times=times, labels=labels, 
                    training=training
                ) if self.classifier_only_cls_token and self.clf_cls_token_type is not None else x

                clf_features_list.append(x)

            x = layers_dict[self.FC](
                clf_features_list, 
                [x] if self.FA in layers_dict else [], 
                cond=clf_cond, 
                training=training
            ) if self.FC in layers_dict else x

            h = layers_dict[self.CAA](
                features_list, 
                cond=clf_cond, 
                training=training
            ) if self.CAA in layers_dict else None

            h = layers_dict[self.CAC](
                clf_features_list, 
                [h] if self.CAA in layers_dict else [], 
                cond=clf_cond, 
                training=training
            ) if self.CAC in layers_dict else h

            x = layers_dict[self.VTB](
                (x, clf_cond), 
                queries=h if self.clf_cross_attention_plug_type == "queries" else None, 
                values=h if self.clf_cross_attention_plug_type == "values" else None, 
                training=training
            ) if self.VTB in layers_dict else x

            x = layers_dict[self.LM](
                (x, clf_cond), 
                training=training
            ) if self.LM in layers_dict else x

            x = layers_dict[self.DS](
                (x, clf_cond), 
                training=training
            ) if self.DS in layers_dict else x

            x = layers_dict[self.US](
                (x, clf_cond), 
                training=training
            ) if self.US in layers_dict else x

            x, x_mean, x_log_var = layers_dict[self.R](
                x, 
                training=training
            ) if self.R in layers_dict else (x, None, None)

            z = layers_dict[self.CTR](
                self.slice_and_flatten_tokens(
                    x, 
                    self.clf_cls_token_regularizer_kwargs["start"], 
                    self.clf_cls_token_regularizer_kwargs["end"]
                ), 
                training=training
            ) if self.CTR in layers_dict else None

            clf_features_list.append(x)
            clf_regs_list.append(z)
            clf_z_vals = (
                x_mean, x_log_var
            ) if x_mean is not None and \
            self.clf_reshaper_ids_dict.get(i+1, "unflatten") == "flatten" else clf_z_vals

        x = self.classifier_feature_extractor(
            x, 
            training=training
        )
        x = self.classifier(
            x, 
            training=training
        )

        return x, clf_cond, clf_features_list, clf_regs_list, clf_z_vals

    def predict_class(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        max_encoder_num: int | None = -1, 
        full_return: bool = False, 
        training: bool | None = None
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor, list[tf.Tensor], 
        list[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
        """Classify inputs while executing only the required main depths.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Image/noisy image
                ``[B,H,W,C]``, integer time IDs ``[B]``, and condition labels
                ``[B]``.
            max_encoder_num (int | None): Main encoder loop stop.  ``None`` uses
                ``self.max_encoder_num``; the default ``-1`` executes all stages.
            full_return (bool): Return classifier intermediates and latent stats.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor | tuple: Class probabilities ``[B,num_classes]`` or, with
            ``full_return=True``, ``(classes, clf_cond, clf_features_list,
            clf_regs_list, clf_z_vals)``.
        """

        max_encoder_num = self.max_encoder_num if max_encoder_num is None else max_encoder_num

        x, cond, features_list, _, _ = self.encode(
            inputs, 
            max_depth=max_encoder_num, 
            training=training
        )
        noises = self.unpatchifier(
            (x, cond), 
            training=training
        ) if self.aggregate_from_noises else None
        x, clf_cond, clf_features_list, clf_regs_list, clf_z_vals = self.compute_class(
            features_list, 
            noises, 
            times=inputs[1], 
            labels=inputs[2], 
            training=training
        )

        # Return classifier features and auxiliary values only when requested.
        if full_return:
            return x, clf_cond, clf_features_list, clf_regs_list, clf_z_vals
        return x

    def add_depths(
        self, 
        depth_spec: str | tuple | set | dict | list | None
    ) -> dict[str, dict[str, int]]:
        """Append transformer and classifier depths through their own APIs.

        An ordinary specification is delegated to ``DiffusionTransformer``
        and therefore grows only ``layers_dicts``. A targeted dictionary may
        contain ``network`` and ``classifier``. The network value uses the
        base transformer's syntax; the classifier value uses the same outer
        list/string/set/dictionary rules and the classifier layer names 
        ``feature_aggregator``, ``feature_connector``,
        ``cross_attention_aggregator``, ``cross_attention_connector``,
        ``vision_transformer_block``, ``reshaper``, ``local_mixer``,
        ``downsampler``, ``upsampler``, and ``cls_token_regularizer``.

        Classifier aggregators read features from the main transformer and
        classifier connectors read preceding classifier depths. The existing
        ID checks and negative/``None`` ID handling are used. The trained
        terminal connector and classifier head are retained, so an appended
        classifier sequence must preserve their feature dimension.

        Args:
            depth_spec (str | tuple | set | dict | list | None): An unscoped
                network specification or a dictionary with optional ``network``
                and ``classifier`` specifications.  For an aggregator/connector,
                a value may be an ID, ID iterable, true/default, or
                ``{"ids": [...]}``.  A transformer block accepts
                ``{"use_decoder": bool, "mlp_output_dim": int | None}`` and a
                reshaper accepts ``{"reshape_type": "flatten" | "unflatten"}``.
                Example: ``{"classifier": [{"feature_connector": {"ids":
                [-1]}}, "vision_transformer_block"]}``.

        Returns:
            dict[str, dict[str, int]]: ``before``, ``added``, and ``after`` depth
            counts for both branches.  An omitted targeted branch reports zero.

        Raises:
            ValueError: If targeted keys/layer names are unknown, the classifier
                terminal stage is not connector-only, or appended layers change
                the feature width expected by an existing head.
        """

        targeted = isinstance(depth_spec, dict) and any(
            name in depth_spec 
            for name in ("network", "classifier")
        )
        # Split targeted growth into denoiser and classifier specifications.
        if targeted:
            # Reject targeted keys outside the two model branches.
            if not all(
                name in ("network", "classifier") for name in depth_spec
            ):
                raise ValueError(
                    "keys in depth_spec dictionary must be in ('network', 'classifier')."
                )

            network_spec = depth_spec.get("network", [])
            classifier_specs = depth_spec.get("classifier", [])
        # Treat an unscoped specification as denoiser-only growth.
        else:
            network_spec = depth_spec
            classifier_specs = []

        classifier_specs = classifier_specs if isinstance(classifier_specs, list) \
                        else [classifier_specs]
        classifier_specs = [
            spec for spec in classifier_specs if spec is not None
        ]

        old_clf_depth = self.clf_depth
        # Delegate directly when no classifier depths were requested.
        if len(classifier_specs) == 0:
            growth = super().add_depths(network_spec)

            # Refresh the encoder limit after denoiser growth used by noise aggregation.
            if self.aggregate_from_noises:
                self.set_max_encoder_num()

            self._init_config[
                "feature_aggregation_ids_dict"
            ] = deepcopy(self.feature_aggregation_ids_dict)
            self._init_config[
                "cross_attention_aggregation_ids_dict"
            ] = deepcopy(self.cross_attention_aggregation_ids_dict)

            # Preserve the base growth result for unscoped requests.
            if not targeted:
                return growth

            return {
                "network": growth["network"], 
                "classifier": {
                    "before": old_clf_depth, 
                    "added": 0, 
                    "after": old_clf_depth, 
                }
            }

        metadata_names = (
            "feature_aggregation_ids_dict", 
            "cross_attention_aggregation_ids_dict", 
            "clf_connection_ids_dict", 
            "clf_cross_attention_ids_dict", "clf_vit_block_ids", 
            "clf_use_decoder_ids", "clf_vit_block_mlp_output_dims", 
            "clf_local_mixer_ids", "clf_downsample_ids", 
            "clf_upsample_ids", "clf_reshaper_ids_dict", 
            "clf_cls_token_regularizer_ids", 
        )
        metadata = {
            name: deepcopy(getattr(self, name)) for name in metadata_names
        }
        old_terminal_key = old_clf_depth + 1
        terminal_layers = dict(self.clf_layers_dicts[-1])

        # Preserve the classifier head as a connector-only terminal depth.
        if set(terminal_layers) != {self.FC}:
            raise ValueError(
                "The terminal classifier depth must contain only its connector."
            )

        terminal_ids = deepcopy(
            self.clf_connection_ids_dict[old_terminal_key]
        )
        old_head_dim = self._get_last_output_dim(
            old_clf_depth, self.clf_layers_dicts, self.clf_dim
        )
        planned_layers = list(self.clf_layers_dicts[:-1])

        try:
            self.clf_connection_ids_dict = {
                key: value for key, value 
                in self.clf_connection_ids_dict.items()
                if key != old_terminal_key
            }
            for layer_spec in classifier_specs:
                # Interpret a string as one enabled classifier layer.
                if isinstance(layer_spec, str):
                    layer_spec = {layer_spec: True}
                # Interpret a collection as several enabled layers in one depth.
                elif isinstance(layer_spec, (tuple, set, frozenset)):
                    layer_spec = dict.fromkeys(layer_spec, True)

                key = len(planned_layers) + 1
                for layer_name, options in layer_spec.items():
                    # Ignore explicitly disabled classifier layers.
                    if options is False:
                        continue
                    
                    # Register source routes for classifier feature handlers.
                    if layer_name in (
                        self.FA[2:], self.FC[2:], 
                        self.CAA[2:], self.CAC[2:]
                    ):
                        ids = options.get("ids") if isinstance(options, dict) else options
                        ids = [-1] if ids is None or ids is True else ids
                        ids = [ids] if isinstance(ids, int) else list(ids)

                        # Route main-network features into the classifier aggregator.
                        if layer_name == self.FA[2:]:
                            dict_name = "feature_aggregation_ids_dict"
                            source_depth = self.depth
                        # Route main-network features into cross-attention aggregation.
                        elif layer_name == self.CAA[2:]:
                            dict_name = "cross_attention_aggregation_ids_dict"
                            source_depth = self.depth
                        # Route earlier classifier features into a residual connector.
                        elif layer_name == self.FC[2:]:
                            dict_name = "clf_connection_ids_dict"
                            source_depth = key-1
                        # Route earlier classifier features into cross-attention.
                        else:
                            dict_name = "clf_cross_attention_ids_dict"
                            source_depth = key-1

                        local_vars = {
                            "depth": source_depth, 
                            dict_name: {key: ids}
                        }
                        self._check_dict_assertions(
                            local_vars, 
                            dict_name, 
                            check_items_num=False, 
                            check_keys=False, 
                            id_less_than_key=False
                        )
                        ids = self._handle_ids(
                            ids, 
                            depth=source_depth, 
                            max_id=source_depth
                        )

                        setattr(self, dict_name, {
                            **getattr(self, dict_name), 
                            key: ids
                        })
                    # Register classifier transformer-block mode and width options.
                    elif layer_name == self.VTB[2:]:
                        block_options = options if isinstance(options, dict) else {}

                        self.clf_vit_block_ids = [
                            *self.clf_vit_block_ids, key
                        ]
                        self.clf_use_decoder_ids = [
                            *self.clf_use_decoder_ids, key
                        ] if block_options.get("use_decoder", False) else self.clf_use_decoder_ids
                        self.clf_vit_block_mlp_output_dims = {
                            **self.clf_vit_block_mlp_output_dims, 
                            key: block_options["mlp_output_dim"]
                        } if block_options.get("mlp_output_dim") is not None else self.clf_vit_block_mlp_output_dims
                    # Register a classifier local mixer at the new depth.
                    elif layer_name == self.LM[2:]:
                        self.clf_local_mixer_ids = [
                            *self.clf_local_mixer_ids, key
                        ]
                    # Register a classifier downsampler at the new depth.
                    elif layer_name == self.DS[2:]:
                        self.clf_downsample_ids = [
                            *self.clf_downsample_ids, key
                        ]
                    # Register a classifier upsampler at the new depth.
                    elif layer_name == self.US[2:]:
                        self.clf_upsample_ids = [
                            *self.clf_upsample_ids, key
                        ]
                    # Register the classifier depth's reshape direction.
                    elif layer_name == self.R[2:]:
                        reshape_type = options.get("reshape_type")
                        self.clf_reshaper_ids_dict = {
                            **self.clf_reshaper_ids_dict, key: reshape_type
                        }
                    # Register an auxiliary classifier token head at the new depth.
                    elif layer_name == self.CTR[2:]:
                        self.clf_cls_token_regularizer_ids = [
                            *self.clf_cls_token_regularizer_ids, key
                        ]
                    # Reject progressive layer names unsupported by the classifier branch.
                    else:
                        raise ValueError(
                            f"Unknown progressive classifier layer: {layer_name}."
                        )

                layers_dict = self._create_clf_layer_dict(
                    key-1, planned_layers
                )
                planned_layers.append(layers_dict)

            new_clf_depth = old_clf_depth + len(classifier_specs)
            # Retarget the terminal connector when it referenced the old last depth.
            if terminal_ids == [old_clf_depth]:
                terminal_ids = [new_clf_depth]

            self.clf_connection_ids_dict = {
                **self.clf_connection_ids_dict, 
                new_clf_depth+1: terminal_ids, 
            }

            # Keep classifier growth compatible with the existing classification head.
            if self._get_last_output_dim(
                len(planned_layers), 
                planned_layers + [terminal_layers], 
                self.clf_dim
            ) != old_head_dim:
                raise ValueError(
                    "Added classifier depths must preserve the classifier-head dimension."
                )

            network_growth = super().add_depths(network_spec)["network"]
        except Exception:
            for name, value in metadata.items():
                setattr(self, name, value)
            raise

        terminal_connector = terminal_layers[self.FC]
        terminal_connector.ids = terminal_ids
        terminal_connector._init_config["ids"] = deepcopy(terminal_ids)

        added_layers = planned_layers[old_clf_depth:]
        current_terminal = self.clf_layers_dicts[-1]
        current_terminal.clear()
        current_terminal.update(added_layers[0])

        self.clf_layers_dicts.extend(added_layers[1:])
        self.clf_layers_dicts.append(terminal_layers)
        self.clf_depth = old_clf_depth + len(added_layers)

        self._save_init_args({
            "clf_depth": self.clf_depth, 
            **{name: getattr(self, name) 
                for name in metadata_names}, 
        })
        connection_ids = {
            key: value for key, value 
            in self.clf_connection_ids_dict.items()
            if key != self.clf_depth + 1
        }
        connection_ids[-1] = terminal_ids
        self._init_config[
            "clf_connection_ids_dict"
        ] = deepcopy(connection_ids)
        self.set_max_encoder_num()

        return {
            "network": network_growth, 
            "classifier": {
                "before": old_clf_depth, 
                "added": self.clf_depth-old_clf_depth, 
                "after": self.clf_depth, 
            }, 
        }


def run_self_tests() -> dict[str, str]:
    """Run deterministic integration tests for every DiTClassifier branch.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"DiTClassifier": "passed"}`` after classifier
        depth, routing, token, pooling, auxiliary, progressive, serialization,
        prediction, and invalid-configuration checks pass.
    """

    tf.keras.backend.clear_session()
    tf.random.set_seed(102)
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
        "depth": 1, 
        "mha_num_heads": 1, 
        "vit_block_mlp_ratio": 1.0, 
        "clf_mha_num_heads": 1, 
        "clf_vit_block_mlp_ratio": 1.0, 
    }


    def make_model(**overrides: object) -> DiTClassifier:
        """Construct a classifier with fresh mutable routing dictionaries.

        Args:
            **overrides (object): Values replacing the CPU-small base config.

        Returns:
            DiTClassifier: A newly initialized test model.
        """

        config = {
            **base,
            "feature_aggregation_ids_dict": {1: (-1,)},
            "clf_connection_ids_dict": {-1: (-1,)},
            **overrides,
        }

        return DiTClassifier(**config)

    # Reproduce aggregate-suite ordering by constructing multiple models from
    # the same public defaults in one process.  Each instance must normalize
    # its own copies without changing the signature-owned dictionaries.
    import inspect


    public_parameters = inspect.signature(DiTClassifier.__init__).parameters
    public_aggregation_default = public_parameters[
        "feature_aggregation_ids_dict"
    ].default
    public_connection_default = public_parameters["clf_connection_ids_dict"].default
    assert public_aggregation_default == {1: (-1,)}
    assert public_connection_default == {-1: (-1,)}
    public_default_first = DiTClassifier(**base)
    public_default_second = DiTClassifier(**base)
    assert public_default_first.feature_aggregation_ids_dict == {1: [1]}
    assert public_default_second.feature_aggregation_ids_dict == {1: [1]}
    assert public_default_first.clf_connection_ids_dict == {2: [1]}
    assert public_default_second.clf_connection_ids_dict == {2: [1]}
    assert (
        public_default_first.feature_aggregation_ids_dict
        is not public_default_second.feature_aggregation_ids_dict
    )
    assert (
        public_default_first.clf_connection_ids_dict
        is not public_default_second.clf_connection_ids_dict
    )
    public_default_first.feature_aggregation_ids_dict[1].append(0)
    public_default_first.clf_connection_ids_dict[2].append(0)
    assert public_default_second.feature_aggregation_ids_dict == {1: [1]}
    assert public_default_second.clf_connection_ids_dict == {2: [1]}
    assert public_aggregation_default == {1: (-1,)}
    assert public_connection_default == {-1: (-1,)}

    model = make_model()
    outputs = model(inputs, full_return=True, training=False)
    assert set(outputs) == {
        "noises", "cond", "features_list", "regs_list", "z_vals",
        "classes", "clf_cond", "clf_features_list", "clf_regs_list",
        "clf_z_vals",
    }
    assert outputs["noises"].shape == (2, 4, 4, 1)
    assert outputs["classes"].shape == (2, 2)
    assert outputs["classes"].dtype == tf.float32
    tf.debugging.assert_near(
        tf.reduce_sum(outputs["classes"], axis=-1), tf.ones((2,)), atol=1e-5
    )
    assert len(model.clf_layers_dicts) == model.clf_depth + 1 == 2
    assert set(model.clf_layers_dicts[-1]) == {model.FC}
    assert model.feature_aggregation_ids_dict == {1: [1]}
    assert model.clf_connection_ids_dict == {2: [1]}
    assert model.max_encoder_num == 1
    assert model.clf_dim == model.first_aggregated_dim == 4
    assert model.clf_mha_num_heads == model.mha_num_heads == 1
    assert model.clf_connection_kwargs == model.connection_kwargs
    assert model.clf_cross_attention_plug_type == model.cross_attention_plug_type
    assert model.clf_drop_prob == model.drop_prob == 0.0
    assert model.clf_drop_per_sample == model.drop_per_sample is True
    assert model.clf_ln_mlp_ratio is None
    assert model.clf_reshaper_ids_dict == {}
    assert model.clf_cls_token_regularizer_ids == []

    inherited = make_model(
        build=False, 
        mha_num_heads=2, 
        vit_block_mlp_ratio=2.0, 
        vit_block_mlp_output_dims={1: 4}, 
        ln_no_adaptation=True, 
        drop_prob=0.2, 
        drop_per_sample=False, 
        connection_kwargs={"connect_type": "add"}, 
        cross_attention_kwargs={"connect_type": "add"}, 
        local_mixer_kwargs={"pos_embed_type": None}, 
        downsample_kwargs={"scaling_method": "avg_pooling"}, 
        upsample_kwargs={"scaling_method": "interpolate"}, 
        cls_token_regularizer_kwargs={"start": 0, "end": 1}, 
        clf_mha_num_heads=None, 
        clf_vit_block_mlp_ratio=None, 
        clf_vit_block_mlp_output_dims=None, 
    )
    assert inherited.clf_connection_kwargs == inherited.connection_kwargs
    assert inherited.clf_cross_attention_kwargs == inherited.cross_attention_kwargs
    assert inherited.clf_mha_num_heads == inherited.mha_num_heads == 2
    assert inherited.clf_vit_block_mlp_ratio == inherited.vit_block_mlp_ratio == 2.0
    assert inherited.clf_vit_block_mlp_output_dims == {1: 4}
    assert inherited.clf_ln_no_adaptation is inherited.ln_no_adaptation is True
    assert inherited.clf_drop_prob == inherited.drop_prob == 0.2
    assert inherited.clf_drop_per_sample is inherited.drop_per_sample is False
    assert inherited.clf_local_mixer_kwargs == inherited.local_mixer_kwargs
    assert inherited.clf_downsample_kwargs == inherited.downsample_kwargs
    assert inherited.clf_upsample_kwargs == inherited.upsample_kwargs
    assert (
        inherited.clf_cls_token_regularizer_kwargs
        == inherited.cls_token_regularizer_kwargs
    )

    overridden = make_model(
        build=False, 
        mha_num_heads=1, 
        vit_block_mlp_ratio=1.0, 
        drop_prob=0.0, 
        drop_per_sample=True, 
        clf_mha_num_heads=2, 
        clf_vit_block_mlp_ratio=3.0, 
        clf_vit_block_mlp_output_dims={}, 
        clf_ln_no_adaptation=True, 
        clf_drop_prob=0.25, 
        clf_drop_per_sample=False, 
        clf_connection_kwargs={}, 
        clf_cross_attention_kwargs={}, 
        clf_local_mixer_kwargs={"pos_embed_type": None}, 
        clf_downsample_kwargs={"scaling_method": "max_pooling"}, 
        clf_upsample_kwargs={"scaling_method": "cnn_transpose"}, 
        clf_cls_token_regularizer_kwargs={"start": 0, "end": 1}, 
    )
    assert overridden.clf_mha_num_heads == 2 != overridden.mha_num_heads
    assert overridden.clf_vit_block_mlp_ratio == 3.0
    assert overridden.clf_vit_block_mlp_output_dims == {}
    assert overridden.clf_ln_no_adaptation is True
    assert overridden.clf_drop_prob == 0.25
    assert overridden.clf_drop_per_sample is False
    assert overridden.clf_connection_kwargs == {}
    assert overridden.clf_cross_attention_kwargs == {}

    assert model.predict_noise(inputs, training=False).shape == (2, 4, 4, 1)
    assert model.predict_class(inputs, training=False).shape == (2, 2)
    class_full = model.predict_class(inputs, full_return=True, training=False)
    assert len(class_full) == 5 and class_full[0].shape == (2, 2)
    model.set_max_encoder_num(0)
    assert model.max_encoder_num == 0
    model.set_max_encoder_num(None)
    assert model.max_encoder_num == 1

    depth_zero = make_model(clf_depth=0, depth=0)
    assert len(depth_zero.layers_dicts) == 0
    assert len(depth_zero.clf_layers_dicts) == 1
    assert depth_zero(inputs, training=False)["classes"].shape == (2, 2)

    for clf_cond_type in (None, "time", "label", "time_label"):
        candidate = make_model(
            clf_cond_type=clf_cond_type, 
            clf_ln_no_adaptation=clf_cond_type is None, 
        )
        result = candidate(inputs, full_return=True, training=False)
        assert result["classes"].shape == (2, 2)
        assert (result["clf_cond"] is None) == (clf_cond_type is None)

    for token_type in (None, "new_weight", "time", "label", "time_label"):
        for force_pool in (False, True):
            candidate = make_model(
                clf_cls_token_type=token_type, 
                force_global_avg_pooling=force_pool, 
                classifier_mlp_ratio=1, 
                dropout_rate=0.25, 
            )
            candidate_output = candidate(inputs, training=False)
            assert candidate_output["classes"].shape == (2, 2)
            expected_pool = force_pool or token_type is None
            assert isinstance(
                candidate.classifier_feature_extractor,
                layers.GlobalAveragePooling1D if expected_pool else layers.Lambda,
            )
            assert len(candidate.classifier.layers) == 3

    shared_token = make_model(
        classifier_only_cls_token=False, 
        cls_token_type="new_weight", 
        clf_cls_token_type=None, 
    )
    assert shared_token.cls_token_type == "new_weight"
    assert shared_token(inputs, training=False)["classes"].shape == (2, 2)

    from_noises = make_model(
        aggregate_from_noises=True, 
        force_global_avg_pooling=True, 
    )
    assert from_noises.max_encoder_num == from_noises.depth
    assert from_noises(inputs, training=False)["classes"].shape == (2, 2)
    assert from_noises.predict_class(inputs, training=False).shape == (2, 2)
    from_noises_full = from_noises.predict_class(
        inputs, full_return=True, training=False
    )
    assert len(from_noises_full) == 5
    assert from_noises_full[0].shape == (2, 2)

    for plug_type in ("values", "queries"):
        cross = make_model(
            feature_aggregation_ids_dict={1: [1]}, 
            cross_attention_aggregation_ids_dict={1: [0]}, 
            cross_attention_aggregation_kwargs={"mlp_output_dim": 4}, 
            clf_cross_attention_plug_type=plug_type, 
            clf_cls_token_type=None, 
            force_global_avg_pooling=True, 
        )
        assert cross(inputs, training=False)["classes"].shape == (2, 2)
        assert cross.cross_attention_aggregation_ids_dict == {1: [0]}

    for plug_type in ("values", "queries"):
        cross_connected = make_model(
            clf_depth=2, 
            clf_connection_ids_dict={2: [0, 1], -1: [-1]}, 
            clf_connection_kwargs={"mlp_output_dim": 4}, 
            clf_cross_attention_ids_dict={2: [0]}, 
            clf_cross_attention_kwargs={"mlp_output_dim": 4}, 
            clf_cross_attention_plug_type=plug_type, 
            clf_vit_block_ids=[1, 2], 
        )
        assert cross_connected(inputs, training=False)["classes"].shape == (2, 2)
        assert cross_connected.clf_connection_ids_dict == {2: [0, 1], 3: [2]}

    additive_aggregation = make_model(
        depth=2, 
        feature_aggregation_ids_dict={1: [0, 1]}, 
        feature_aggregation_kwargs={"connect_type": "add"}, 
    )
    assert additive_aggregation.feature_aggregation_ids_dict == {1: [0, 1]}
    assert additive_aggregation(inputs, training=False)["classes"].shape == (2, 2)

    expanded_aggregation = make_model(
        depth=2, 
        feature_aggregation_ids_dict={1: [None]}, 
    )
    assert expanded_aggregation.feature_aggregation_ids_dict == {1: [0, 1, 2]}
    assert expanded_aggregation.first_aggregated_dim == 12
    assert expanded_aggregation(inputs, training=False)["classes"].shape == (2, 2)

    from diffusion.layers.block.di_t_decoder_block import DiTDecoderBlock

    explicit_classifier_block = make_model(
        clf_dim=4, 
        clf_dim_forced=True, 
        clf_use_decoder_ids=[1], 
        clf_mha_key_dim=2, 
        clf_mha_value_dim=3, 
        clf_mha_num_heads=2, 
        clf_vit_block_mlp_output_dims={1: 4}, 
        clf_drop_prob=0.5, 
        clf_drop_per_sample=False, 
        classifier_mlp_ratio=2, 
        classifier_mlp_activation_func="relu", 
        dropout_rate=0.25, 
    )
    explicit_block = explicit_classifier_block.clf_layers_dicts[0][
        explicit_classifier_block.VTB
    ]
    assert isinstance(explicit_block, DiTDecoderBlock)
    assert explicit_classifier_block.clf_use_decoder_ids == [1]
    assert explicit_block.key_dim == 2
    assert explicit_block.value_dim == 3
    assert explicit_block.mlp_output_dim == 4
    assert explicit_block.drop_prob == 0.5
    assert explicit_block.drop_per_sample is False
    explicit_training_output = explicit_classifier_block(inputs, training=True)
    assert explicit_training_output["classes"].shape == (2, 2)
    assert bool(tf.reduce_all(tf.math.is_finite(explicit_training_output["classes"])))
    assert explicit_classifier_block.classifier.layers[0].activation.__name__ == "relu"

    policy = make_model(
        name="policy_classifier", 
        name_prefix="policy/", 
        dtype="float64", 
        trainable=False, 
        dynamic=True, 
    )
    policy_output = policy(inputs, training=False)
    assert policy.name == "policy_classifier"
    assert policy.name_prefix == "policy/"
    assert policy.dtype_policy.name == "float64"
    assert policy.dynamic is True
    assert policy.trainable is False
    assert policy_output["noises"].dtype == tf.float32
    assert policy_output["classes"].dtype == tf.float32

    scaled = make_model(
        clf_depth=2, 
        clf_vit_block_ids=[], 
        clf_downsample_ids=[1], 
        clf_upsample_ids=[2], 
        clf_downsample_kwargs={"scaling_method": "avg_pooling"}, 
        clf_upsample_kwargs={"scaling_method": "interpolate"}, 
    )
    assert scaled(inputs, training=False)["classes"].shape == (2, 2)
    mixed = make_model(
        clf_local_mixer_ids=[1], 
        clf_local_mixer_kwargs={"pos_embed_type": None}, 
    )
    assert mixed(inputs, training=False)["classes"].shape == (2, 2)

    for add_kl in (False, True):
        reshaped = make_model(
            clf_depth=2, 
            clf_vit_block_ids=[], 
            clf_reshaper_ids_dict={1: "flatten", 2: "unflatten"}, 
            clf_reshaper_kwargs={"add_kl": add_kl, "latent_dim_ratio": 1.0}, 
            force_global_avg_pooling=True, 
        )
        reshaped_outputs = reshaped(inputs, full_return=True, training=False)
        assert reshaped_outputs["classes"].shape == (2, 2)
        latent = reshaped_outputs["clf_z_vals"]
        assert latent[0] is not None and latent[1] is not None

    regularized = make_model(
        clf_cls_token_regularizer_ids=[None], 
        cls_token_regularizer_ids=[None], 
    )
    regularized_outputs = regularized(inputs, full_return=True, training=False)
    assert all(
        item.shape == (2, 2)
        for item in regularized_outputs["clf_regs_list"][:2]
    )
    assert regularized_outputs["clf_regs_list"][-1] is None
    assert all(item.shape == (2, 2) for item in regularized_outputs["regs_list"])

    progressive = make_model(clf_depth=1)
    growth = progressive.add_depths({
        "network": "vision_transformer_block", 
        "classifier": "vision_transformer_block", 
    })
    assert growth["network"] == {"before": 1, "added": 1, "after": 2}
    assert growth["classifier"] == {"before": 1, "added": 1, "after": 2}
    assert progressive(inputs, training=False)["classes"].shape == (2, 2)
    no_growth = progressive.add_depths({"network": [], "classifier": []})
    assert no_growth["network"]["added"] == no_growth["classifier"]["added"] == 0
    clone = DiTClassifier.from_config(progressive.get_config())
    assert clone.depth == 2 and clone.clf_depth == 2
    assert clone(inputs, training=False)["classes"].shape == (2, 2)

    progressive_components = make_model(
        clf_dim=4, 
        clf_dim_forced=True, 
        clf_cls_token_type=None, 
        force_global_avg_pooling=True, 
    )
    component_growth = progressive_components.add_depths({
        "classifier": [
            {"feature_aggregator": {"ids": [-1]}}, 
            {"feature_connector": {"ids": [-1]}}, 
            {
                "cross_attention_aggregator": {"ids": [-1]}, 
                "vision_transformer_block": True, 
            }, 
            {
                "cross_attention_connector": {"ids": [-1]}, 
                "vision_transformer_block": True, 
            },
            "local_mixer", 
            "downsampler", 
            "upsampler", 
            {"reshaper": {"reshape_type": "flatten"}}, 
            {"reshaper": {"reshape_type": "unflatten"}}, 
            "cls_token_regularizer", 
            {
                "vision_transformer_block": {
                    "use_decoder": True, 
                    "mlp_output_dim": 4, 
                }
            }, 
            {"vision_transformer_block": False}, 
        ]
    })
    assert component_growth["network"] == {
        "before": 1, 
        "added": 0, 
        "after": 1, 
    }
    assert component_growth["classifier"] == {
        "before": 1, 
        "added": 12, 
        "after": 13, 
    }
    assert progressive_components.feature_aggregation_ids_dict[2] == [1]
    assert progressive_components.clf_connection_ids_dict[3] == [2]
    assert progressive_components.cross_attention_aggregation_ids_dict[4] == [1]
    assert progressive_components.clf_cross_attention_ids_dict[5] == [4]
    assert progressive_components.clf_local_mixer_ids == [6]
    assert progressive_components.clf_downsample_ids == [7]
    assert progressive_components.clf_upsample_ids == [8]
    assert progressive_components.clf_reshaper_ids_dict == {
        9: "flatten", 
        10: "unflatten", 
    }
    assert progressive_components.clf_cls_token_regularizer_ids == [11]
    assert progressive_components.clf_use_decoder_ids == [12]
    assert progressive_components.clf_vit_block_mlp_output_dims == {12: 4}
    assert progressive_components.clf_layers_dicts[12] == {}
    component_outputs = progressive_components(
        inputs, full_return=True, training=False
    )
    assert component_outputs["classes"].shape == (2, 2)
    assert component_outputs["clf_regs_list"][11].shape == (2, 2)

    progressive_noop = make_model()
    for empty_classifier_spec in (None, []):
        noop_growth = progressive_noop.add_depths({
            "network": [], 
            "classifier": empty_classifier_spec, 
        })
        assert noop_growth["network"]["added"] == 0
        assert noop_growth["classifier"] == {
            "before": 1, 
            "added": 0, 
            "after": 1, 
        }

    for collection_specification in (
        ("local_mixer", "vision_transformer_block"), 
        {"local_mixer", "vision_transformer_block"}, 
        frozenset({"local_mixer", "vision_transformer_block"}), 
    ):
        collection_classifier = make_model(
            clf_cls_token_type=None, 
            force_global_avg_pooling=True, 
        )
        collection_result = collection_classifier.add_depths({
            "classifier": collection_specification, 
        })
        assert collection_result["classifier"] == {
            "before": 1, 
            "added": 1, 
            "after": 2, 
        }
        assert collection_classifier.clf_local_mixer_ids == [2]
        assert collection_classifier.clf_vit_block_ids == [1, 2]
        assert collection_classifier(inputs, training=False)["classes"].shape == (
            2, 2
        )

    normalized_handlers = make_model(
        clf_dim=4, 
        clf_dim_forced=True, 
        clf_cls_token_type=None, 
        force_global_avg_pooling=True, 
    )
    handler_growth = normalized_handlers.add_depths({
        "classifier": [
            "feature_aggregator", 
            {"feature_connector": None}, 
            {
                "cross_attention_aggregator": 0, 
                "vision_transformer_block": True, 
            }, 
            {
                "cross_attention_connector": True, 
                "vision_transformer_block": True, 
            }, 
        ]
    })
    assert handler_growth["classifier"] == {
        "before": 1, 
        "added": 4, 
        "after": 5, 
    }
    assert normalized_handlers.feature_aggregation_ids_dict[2] == [1]
    assert normalized_handlers.clf_connection_ids_dict[3] == [2]
    assert normalized_handlers.cross_attention_aggregation_ids_dict[4] == [0]
    assert normalized_handlers.clf_cross_attention_ids_dict[5] == [4]
    assert normalized_handlers(inputs, training=False)["classes"].shape == (2, 2)

    terminal_error = make_model()
    terminal_stage = terminal_error.clf_layers_dicts[-1]
    terminal_stage["extra"] = terminal_stage[terminal_error.FC]
    try:
        terminal_error.add_depths({"classifier": "vision_transformer_block"})
    except ValueError as error:
        assert "terminal classifier depth must contain only its connector" in str(
            error
        )
    else:
        raise AssertionError("A non-connector terminal classifier stage must fail")

    classifier_rollback = make_model()
    classifier_rollback.clf_layers_dicts[-1][
        classifier_rollback.FC
    ].output_dim = None
    rollback_depth = classifier_rollback.clf_depth
    rollback_layers_count = len(classifier_rollback.clf_layers_dicts)
    rollback_metadata_names = (
        "feature_aggregation_ids_dict", 
        "cross_attention_aggregation_ids_dict", 
        "clf_connection_ids_dict", 
        "clf_cross_attention_ids_dict", 
        "clf_vit_block_ids", 
        "clf_use_decoder_ids", 
        "clf_vit_block_mlp_output_dims", 
        "clf_local_mixer_ids", 
        "clf_downsample_ids", 
        "clf_upsample_ids", 
        "clf_reshaper_ids_dict", 
        "clf_cls_token_regularizer_ids", 
    )
    rollback_metadata = {
        name: deepcopy(getattr(classifier_rollback, name))
        for name in rollback_metadata_names
    }
    try:
        classifier_rollback.add_depths({
            "classifier": {
                "vision_transformer_block": {"mlp_output_dim": 6}
            }
        })
    except ValueError as error:
        assert "preserve the classifier-head dimension" in str(error)
    else:
        raise AssertionError("A progressive classifier head-width change must fail")
    assert classifier_rollback.clf_depth == rollback_depth
    assert len(classifier_rollback.clf_layers_dicts) == rollback_layers_count
    assert all(
        getattr(classifier_rollback, name) == rollback_metadata[name]
        for name in rollback_metadata_names
    )
    assert classifier_rollback(inputs, training=False)["classes"].shape == (2, 2)

    parser_rollback = make_model()
    parser_depth = parser_rollback.clf_depth
    parser_layers_count = len(parser_rollback.clf_layers_dicts)
    parser_metadata = {
        name: deepcopy(getattr(parser_rollback, name))
        for name in rollback_metadata_names
    }
    try:
        parser_rollback.add_depths({
            "classifier": ["local_mixer", "unknown"]
        })
    except ValueError as error:
        assert "Unknown progressive classifier layer" in str(error)
    else:
        raise AssertionError("A later unknown classifier component must fail")
    assert parser_rollback.clf_depth == parser_depth
    assert len(parser_rollback.clf_layers_dicts) == parser_layers_count
    assert all(
        getattr(parser_rollback, name) == parser_metadata[name]
        for name in rollback_metadata_names
    )

    invalid_cases = (
        {"use_cfg": False}, 
        {"aggregate_from_noises": True, "use_unpatchify": False}, 
        {"feature_aggregation_ids_dict": {2: [0]}}, 
        {"clf_connection_ids_dict": {1: [0]}}, 
        {"clf_dim_forced": True, "clf_dim": None}, 
        {"clf_cross_attention_plug_type": "unknown"}, 
        {"feature_aggregation_kwargs": {"unknown": 1}}, 
        {"cross_attention_aggregation_kwargs": {"unknown": 1}}, 
        {"clf_connection_kwargs": {"unknown": 1}}, 
        {"clf_local_mixer_kwargs": {"unknown": 1}}, 
        {"clf_downsample_kwargs": {"unknown": 1}}, 
        {"clf_upsample_kwargs": {"unknown": 1}}, 
        {"clf_reshaper_kwargs": {"unknown": 1}}, 
        {"clf_cls_token_regularizer_kwargs": {"unknown": 1}}, 
    )
    for overrides in invalid_cases:
        try:
            make_model(build=False, **overrides)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Expected invalid classifier config: {overrides}")
    try:
        progressive.add_depths({"classifier": "unknown"})
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown progressive classifier layers must fail")
    try:
        progressive.add_depths({"network": [], "classifier": [], "bad": []})
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown targeted progressive keys must fail")

    tf.keras.backend.clear_session()
    return {"DiTClassifier": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
