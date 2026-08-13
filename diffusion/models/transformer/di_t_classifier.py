import tensorflow as tf
from tensorflow.keras import layers, models

from copy import deepcopy

from typing import Literal

from . import CondType, TokenType, IdsType, IdsDictType

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer


class DiTClassifier(DiffusionTransformer):
    """
    
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
        clf_vit_block_mlp_output_dims: dict[int] | None = None, 
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
        clf_reshaper_ids_dict: dict[int] = {}, 
        clf_reshaper_kwargs: dict = {}, 
        clf_cls_token_regularizer_ids: IdsType = [], 
        clf_cls_token_regularizer_kwargs: dict | None = None, 
        force_global_avg_pooling: bool = False, 
        classifier_mlp_ratio: int | None = None, 
        classifier_mlp_activation_func: str = "tanh", 
        dropout_rate: float = 0., 
        build: bool = True, 
        **kwargs
    ):
        super().__init__(
            cls_token_type=None if classifier_only_cls_token and \
                        (temp_val:=kwargs.pop("cls_token_type", None)) \
                        is not None else temp_val, 
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
        self.clf_dim = self.first_aggregated_dim if self.clf_dim is None else self.clf_dim
        self.clf_grid_size = self._get_ids_grid_size(
            ids_set=self.feature_aggregation_ids_dict[1], 
            layers_dicts=self.layers_dicts, 
            base_grid_size=self.grid_size, 
            must_be_same=True
        )
        self.clf_connection_ids_dict[self.clf_depth+1] = self.clf_connection_ids_dict.pop(-1, (-1,))

        self._create_clf_embedders()
        self.cls_token = self._create_cls_token(
            self.clf_dim, 
            self.cls_token_pos_merger_type, 
            self.cls_token_freq_dim, 
            self.cls_token_mlp_ratio, 
            self.clf_cls_token_type, 
            "clf_"
        ) if self.classifier_only_cls_token else self.cls_token
        self._create_clf_layers()

        if self.force_global_avg_pooling or not (
        (not self.classifier_only_cls_token and \
        self.cls_token_type is not None) or \
        (self.classifier_only_cls_token and \
        self.clf_cls_token_type is not None)):
            self.classifier_feature_extractor = layers.GlobalAveragePooling1D(
                name=f"{self.name_prefix}classifier_feature_extractor"
            )
        else:
            self.classifier_feature_extractor = layers.Lambda(
                lambda x: x[:, 0, :], 
                name=f"{self.name_prefix}classifier_feature_extractor"
            )

        self.classifier = models.Sequential( # TODO: build it as a functional model
            name=f"{self.name_prefix}classes"
        )
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

        if self.build_:
            self.build()

    def _check_clf_assertions(self, local_vars: dict):
        local_vars["depth"] = self.depth

        assert self.use_cfg, \
            "use_cfg must be True for classification to work."

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
                        "clf_cls_token_regularizer_ids"]):
        for name, clf_part_value in local_vars.items():
            if not name.startswith("clf_") or name in exclude:
                continue

            noise_part_name = name.replace("clf_", '')
            noise_part_value = getattr(self, noise_part_name)
            clf_part_value = noise_part_value if clf_part_value is None else clf_part_value

            setattr(self, name, clf_part_value)

        if self.clf_dim_forced:
            assert self.clf_dim is not None, \
                "When clf_dim_forced is true, clf_dim cannot be None."

    def _handle_all_clf_ids(self):
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
                                        skip_reshaper: bool):
        last_output_dim = None

        if (key:=self.FA) in layers_dict:
            last_output_dim = layers_dict[key].output_dim

        last_output_dim = last_output_dim if (last_output_dim_:=
            super()._get_layers_dict_last_output_dim(
                layers_dict, 
                skip_reshaper
        )) is None else last_output_dim_

        return last_output_dim

    def _create_clf_embedders(self):
        self._clf_cond_type = self.clf_cond_type if self.clf_cond_type is not None and not self.clf_ln_no_adaptation else []
        self._clf_cls_token_type = self.clf_cls_token_type if self.clf_cls_token_type is not None else []

        clf_embed_times_flag = "time" in self._clf_cls_token_type or "time" in self._clf_cond_type
        clf_embed_labels_flag = "label" in self._clf_cls_token_type or "label" in self._clf_cond_type
        clf_conds_merger_flag = ("time" in self._clf_cls_token_type and "label" in self._clf_cls_token_type
                                ) or ("time" in self._clf_cond_type and "label" in self._clf_cond_type)

        if self.classifier_only_cls_token:
            if flag1:=("time" in self._cls_token_type) and not clf_embed_times_flag:
                self.time_embedder = None
            if flag2:=("label" in self._cls_token_type) and not clf_embed_labels_flag:
                self.label_embedder = None
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

    def _create_clf_layer_dict(self, i: int, layers_dicts: list[dict]):
        layers_dict = {}
        key = i+1

        if key in self.feature_aggregation_ids_dict:
            layers_dict[self.FA] = self._create_feature_handler(
                ids_set=self.feature_aggregation_ids_dict[key], 
                layers_dicts=self.layers_dicts, 
                base_dim=self.clf_dim, 
                dim_forced=self.clf_dim_forced, 
                ln_mlp_ratio=self.clf_ln_mlp_ratio, 
                ln_no_adaptation=self.clf_ln_no_adaptation, 
                zero_index_base_dim=self.dim, 
                increased_dim=self._get_last_output_dim(
                    i=i-1, 
                    layers_dicts=layers_dicts, 
                    base_dim=self.clf_dim
                ) if key not in self.clf_connection_ids_dict and i != 0 else 0, 
                output_dim_flag=key not in self.clf_connection_ids_dict, 
                kwargs=self.feature_aggregation_kwargs, 
                name=f"{self.name_prefix}clf_depth_{key}_{self.FA[2:]}"
            )

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
                ), 
                kwargs=self.clf_connection_kwargs, 
                name=f"{self.name_prefix}clf_depth_{key}_{self.FC[2:]}"
            )

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

        if key in self.clf_cls_token_regularizer_ids:
            layers_dict[self.CTR] = self._create_token_regularizer(
                name=f"{self.name_prefix}clf_depth_{key}_{self.CTR[2:]}"
            )

        return layers_dict

    def _create_clf_layers(self):
        self.clf_layers_dicts = []

        for i in range(self.clf_depth+1):
            self.clf_layers_dicts.append(
                self._create_clf_layer_dict(i, self.clf_layers_dicts)
            )

    def _register_clf_depth_layer_spec(self, layer_spec, key: int):
        aliases = {
            "feature_aggregation": "feature_aggregator", 
            "aggregation": "feature_aggregator", 
            "connection": "feature_connector", 
            "cross_attention_aggregation": "cross_attention_aggregator", 
            "cross_attention": "cross_attention_connector", 
            "vit_block": "vision_transformer_block", 
            "encoder_block": "vision_transformer_block", 
            "downsample": "downsampler", 
            "upsample": "upsampler", 
        }
        allowed = {
            "feature_aggregator", "feature_connector", 
            "cross_attention_aggregator", 
            "cross_attention_connector", 
            "vision_transformer_block", "decoder_block", 
            "local_mixer", "downsampler", "upsampler", 
            "reshaper", "cls_token_regularizer", 
        }

        for layer_name, options in self._normalize_depth_layer_spec(
            layer_spec
        ).items():
            layer_name = aliases.get(layer_name, layer_name)
            assert layer_name in allowed, \
                f"Unknown progressive classifier layer: {layer_name}."
            if options is False:
                continue

            ids = options.get("ids") \
                  if isinstance(options, dict) else options
            if layer_name == "feature_aggregator":
                self.feature_aggregation_ids_dict[key] = \
                    self._resolve_progressive_ids(ids, self.depth+1)
            elif layer_name == "feature_connector":
                self.clf_connection_ids_dict[key] = \
                    self._resolve_progressive_ids(ids, key)
            elif layer_name == "cross_attention_aggregator":
                self.cross_attention_aggregation_ids_dict[key] = \
                    self._resolve_progressive_ids(ids, self.depth+1)
            elif layer_name == "cross_attention_connector":
                self.clf_cross_attention_ids_dict[key] = \
                    self._resolve_progressive_ids(ids, key)
            elif layer_name in (
                "vision_transformer_block", "decoder_block"
            ):
                self.clf_vit_block_ids = [*self.clf_vit_block_ids, key]
                block_options = options if isinstance(options, dict) else {}
                if layer_name == "decoder_block" or \
                block_options.get("use_decoder", False):
                    self.clf_use_decoder_ids = [
                        *self.clf_use_decoder_ids, key
                    ]
                if block_options.get("mlp_output_dim") is not None:
                    self.clf_vit_block_mlp_output_dims[key] = \
                        block_options["mlp_output_dim"]
            elif layer_name == "local_mixer":
                self.clf_local_mixer_ids = [*self.clf_local_mixer_ids, key]
            elif layer_name == "downsampler":
                self.clf_downsample_ids = [*self.clf_downsample_ids, key]
            elif layer_name == "upsampler":
                self.clf_upsample_ids = [*self.clf_upsample_ids, key]
            elif layer_name == "reshaper":
                reshape_type = options.get("reshape_type") \
                               if isinstance(options, dict) else options
                assert reshape_type in ("flatten", "unflatten"), \
                    "A progressive classifier reshaper must be flatten "\
                    "or unflatten."
                self.clf_reshaper_ids_dict[key] = reshape_type
            elif layer_name == "cls_token_regularizer":
                self.clf_cls_token_regularizer_ids = [
                    *self.clf_cls_token_regularizer_ids, key
                ]

    def _update_clf_depth_config(self):
        for name in self._clf_depth_metadata_names():
            self._init_config[name] = deepcopy(getattr(self, name))

        connection_ids = deepcopy(self.clf_connection_ids_dict)
        connection_ids[-1] = connection_ids.pop(self.clf_depth+1)
        self._init_config["clf_connection_ids_dict"] = connection_ids

    def _clf_depth_metadata_names(self):
        return (
            "clf_depth", "feature_aggregation_ids_dict", 
            "cross_attention_aggregation_ids_dict", 
            "clf_connection_ids_dict", 
            "clf_cross_attention_ids_dict", "clf_vit_block_ids", 
            "clf_use_decoder_ids", "clf_vit_block_mlp_output_dims", 
            "clf_local_mixer_ids", "clf_downsample_ids", 
            "clf_upsample_ids", "clf_reshaper_ids_dict", 
            "clf_cls_token_regularizer_ids", 
        )

    def _append_classifier_depths(self, depth_specs):
        depth_specs = depth_specs if isinstance(depth_specs, list) \
                      else [depth_specs]
        depth_specs = [spec for spec in depth_specs if spec is not None]
        if len(depth_specs) == 0:
            return 0

        metadata_names = self._clf_depth_metadata_names()[1:]
        metadata = {
            name: deepcopy(getattr(self, name)) for name in metadata_names
        }
        old_depth = self.clf_depth
        old_terminal_key = old_depth+1
        terminal_layers = dict(self.clf_layers_dicts[-1])
        assert set(terminal_layers) == {self.FC}, \
            "The terminal classifier depth must only contain its feature "\
            "connector."
        old_terminal_input_dim = self._get_last_output_dim(
            old_depth-1, self.clf_layers_dicts, self.clf_dim
        )
        old_head_dim = self._get_last_output_dim(
            old_depth, self.clf_layers_dicts, self.clf_dim
        )
        planned_layers = list(self.clf_layers_dicts[:-1])
        added_layers = []

        try:
            terminal_ids = self.clf_connection_ids_dict.pop(
                old_terminal_key
            )
            for layer_spec in depth_specs:
                key = len(planned_layers) + 1
                self._register_clf_depth_layer_spec(layer_spec, key)
                layers_dict = self._create_clf_layer_dict(
                    key-1, planned_layers
                )
                assert len(layers_dict) > 0, \
                    f"No layer was enabled for added classifier depth {key}."
                planned_layers.append(layers_dict)
                added_layers.append(layers_dict)

            new_depth = old_depth + len(added_layers)
            if terminal_ids == [old_depth]:
                terminal_ids = [new_depth]
            self.clf_connection_ids_dict[new_depth+1] = terminal_ids

            output_dim = self._get_last_output_dim(
                len(planned_layers)-1, planned_layers, self.clf_dim
            )
            assert output_dim == old_terminal_input_dim, \
                "Added classifier depths must preserve the feature "\
                "dimension expected by the terminal connector."
            final_layers = planned_layers + [terminal_layers]
            assert self._get_last_output_dim(
                len(final_layers)-1, final_layers, self.clf_dim
            ) == old_head_dim, \
                "Added classifier depths must preserve the feature "\
                "dimension used by the classifier head."
        except Exception:
            for name, value in metadata.items():
                setattr(self, name, value)
            raise

        terminal_connector = terminal_layers[self.FC]
        terminal_connector.ids = terminal_ids
        terminal_connector._init_config["ids"] = deepcopy(terminal_ids)

        current_terminal = self.clf_layers_dicts[-1]
        current_terminal.clear()
        current_terminal.update(added_layers[0])
        self.clf_layers_dicts.extend(added_layers[1:])
        self.clf_layers_dicts.append(terminal_layers)
        self.clf_depth = old_depth + len(added_layers)
        self._update_clf_depth_config()
        self.set_max_encoder_num()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

        return self.clf_depth - old_depth

    def add_depths(self, depth_spec):
        """Append network and/or classifier layer dictionaries."""

        targeted = isinstance(depth_spec, dict) and any(
            name in depth_spec for name in ("network", "classifier")
        )
        if not targeted:
            return super().add_depths(depth_spec)

        unknown_targets = set(depth_spec) - {"network", "classifier"}
        assert not unknown_targets, \
            f"Unknown progressive depth targets: {unknown_targets}."

        with tf.device("/CPU:0"):
            probe = self.__class__.from_config(deepcopy(self.get_config()))
            probe._add_depths(depth_spec)
        del probe

        return self._add_depths(depth_spec)

    def _add_depths(self, depth_spec):
        targeted = isinstance(depth_spec, dict) and any(
            name in depth_spec for name in ("network", "classifier")
        )
        if not targeted:
            return super()._add_depths(depth_spec)

        unknown_targets = set(depth_spec) - {"network", "classifier"}
        assert not unknown_targets, \
            f"Unknown progressive depth targets: {unknown_targets}."

        old_depth = self.depth
        old_clf_depth = self.clf_depth
        network_added = self._append_network_depths(
            depth_spec.get("network", [])
        )
        classifier_added = self._append_classifier_depths(
            depth_spec.get("classifier", [])
        )

        return {
            "network": {
                "before": old_depth, 
                "added": network_added, 
                "after": self.depth, 
            }, 
            "classifier": {
                "before": old_clf_depth, 
                "added": classifier_added, 
                "after": self.clf_depth, 
            }, 
        }

    def set_max_encoder_num(self, max_encoder_num: int | None = None):
        aggregation_ids = []
        [aggregation_ids.extend(value) for value in \
            self.feature_aggregation_ids_dict.values()]
        [aggregation_ids.extend(value) for value in \
            self.cross_attention_aggregation_ids_dict.values()]

        self.max_encoder_num = max(
            aggregation_ids
        ) if max_encoder_num is None else max_encoder_num

    def predict_noise(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        full_return: bool = False, 
        training: bool | None = None
    ):
        outputs = super().call(
            inputs, 
            full_return=full_return, 
            training=training
        )

        return outputs

    def compute_class(
        self, 
        features_list: list[tf.Tensor], 
        times: tf.Tensor, 
        labels: tf.Tensor, 
        training: bool | None = None
    ) -> tuple[tf.Tensor, tf.Tensor, list[tf.Tensor], 
        list[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
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
            if i == 0:
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
        max_encoder_num = self.max_encoder_num if max_encoder_num is None else max_encoder_num

        *_, features_list, _, _ = self.encode(
            inputs, 
            max_depth=max_encoder_num, 
            training=training
        )
        x, clf_cond, clf_features_list, clf_regs_list, clf_z_vals = self.compute_class(
            features_list, 
            times=inputs[1], 
            labels=inputs[2], 
            training=training
        )

        if full_return:
            return x, clf_cond, clf_features_list, clf_regs_list, clf_z_vals
        return x

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        full_return: bool = False, 
        training: bool | None = None
    ) -> dict:
        noises, cond, features_list, regs_list, z_vals = super().call(
            inputs, 
            full_return=True, 
            training=training
        )
        outputs = self.compute_class(
            features_list, 
            times=inputs[1], 
            labels=inputs[2], 
            training=training
        )
        output_dict = {
            "noises": noises, 
        }

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
        else:
            output_dict["classes"] = outputs

        return output_dict
