from typing import Literal

import tensorflow as tf
from tensorflow.keras import layers, models

from common.argument_saver import ArgumentSaverModel

from autoencoder.variational_autoencoder import VariationalAutoencoder

from . import CondType, TokenType, IdsType, IdsDictType

from diffusion.layers import MergeType
from diffusion.layers.block.vision_transformer_block import VisionTransformerBlock
from diffusion.layers.block.di_t_decoder_block import DiTDecoderBlock
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
    """

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
        num_classes: int = 10, 
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
        vit_block_mlp_output_dims: dict[int] = {}, 
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
        reshaper_ids_dict: dict[int] = {}, 
        cls_token_regularizer_ids: IdsType = [], 
        cls_token_regularizer_kwargs: dict = {"start": 0, "end": 1}, 
        final_ffn_activation_func: str = "linear", 
        use_final_cnn: bool = False, 
        final_cnn_hidden_dim: int | None = None, 
        final_cnn_residual: bool = True, 
        final_activation_func: str = "linear", 
        use_unpatchify: bool = True, 
        name_prefix: str = "", 
        build: bool = True, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._check_assertions(locals())
        self._save_init_args(locals())
        self._handle_all_ids()

        self.num_labels = self.num_classes + int(self.use_cfg)
        self.grid_size = self.image_size // self.patch_size
        self.patches_dim = self.dim
        self.cond_dim = self.dim if self.cond_dim is None else self.cond_dim
        self.dim = self.patches_dim + self.cond_dim if self.patches_conds_merger_type == "concat" else self.dim
        self.cond_embedder_dim = self.cond_dim // 2 if self.conds_merger_type == "concat" and \
                                self.cond_type == "time_label" else self.cond_dim

        self._create_embedders()
        self.cls_token = self._create_cls_token(
            self.dim, 
            self.cls_token_pos_merger_type, 
            self.cls_token_freq_dim, 
            self.cls_token_mlp_ratio, 
            self.cls_token_type
        ) if self.cls_token_type is not None else None
        self._create_layers()

        if self.use_unpatchify: # TODO: make it one functional model
            self.final_layer_norm = AdaLNZero(
                dim=self._get_unforced_total_dim(
                    [self.depth], 
                    self.layers_dicts, 
                    self.dim
                ), 
                return_gate=False, 
                mlp_ratio=self.ln_mlp_ratio, 
                no_adaptation=self.ln_no_adaptation, 
                name=f"{self.name_prefix}final_layer_norm"
            )
            self.final_ffn = layers.Dense(
                self.patch_size * self.patch_size * self.channels, 
                kernel_initializer="zeros", 
                activation=self.final_ffn_activation_func, 
                name=f"{self.name_prefix}final_ffn"
            )

            if self.use_final_cnn:
                hidden_dim = self.dim if self.final_cnn_hidden_dim is None \
                            else self.final_cnn_hidden_dim

                self.final_cnn = models.Sequential([
                    layers.Conv2D(
                        hidden_dim, 
                        kernel_size=3, 
                        padding="same", 
                        activation="swish", 
                        name="refine_conv_1"
                    ),
                    layers.Conv2D(
                        self.channels, 
                        kernel_size=3, 
                        padding="same", 
                        kernel_initializer="zeros", 
                        bias_initializer="zeros", 
                        name="refine_conv_2"
                    ),
                ], name=f"{self.name_prefix}final_cnn_refiner")         

            self.final_activation = layers.Activation(
                self.final_activation_func, 
                name=f"{self.name_prefix}noises"
            )

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
        none_is_filler: bool = True, 
    ):
        if check_items_num:
            assert local_vars[depth_name] == 0 or \
                len(local_vars[dict_name]) <= local_vars[depth_name], \
                f"Items (id sets) in {dict_name} cannot be more than {depth_name}."

        if not isinstance((dict_:=local_vars[dict_name]), dict):
            dict_ = {1: dict_}

        for key, value in dict_.items():
            if check_keys:
                if len(allowed_keys) == 0:
                    assert local_vars[depth_name] == 0 or \
                        1 <= key <= local_vars[depth_name], \
                        f"Keys in {dict_name} need to be in [1, {local_vars[depth_name]}] range."
                else:
                    assert key in allowed_keys, \
                        f"Keys in {dict_name} need to be one of {allowed_keys}."

            if check_values:
                for id_ in value:
                    if id_less_than_key:
                        assert (none_is_filler and id_ is None) or id_ < key, \
                            f"The ids in each set of {dict_name} can only be less than their key."

                    if len(allowed_values) == 0:
                        assert local_vars[second_depth_name] == 0 or \
                            (none_is_filler and id_ is None) or \
                            -(local_vars[second_depth_name]+1) <= id_ <= local_vars[second_depth_name], \
                            f"The ids in each set of {dict_name} can only be None or in "\
                            f"[-{second_depth_name}-1, {second_depth_name}] range."
                    else:
                        assert local_vars[second_depth_name] == 0 or \
                        id_ in allowed_values, \
                        f"The ids in each set of {dict_name} can only be one of {allowed_values} ."

    def _check_assertions(self, local_vars: dict):
        assert local_vars["image_size"] % local_vars["patch_size"] == 0, \
            "image_size must be divisible by patch_size."

        if local_vars["patches_conds_merger_type"] == "add":
            assert local_vars["dim"] == local_vars["cond_dim"], \
                "When patches_conds_merger_type is add, dim and cond_dim must be equal."

        if local_vars["cond_type"] is None:
            assert local_vars["ln_no_adaptation"], \
                "When cond_type is None, layer_norm cannot use adaptation."

        assert local_vars["cls_token_type"] in (vals:=(None, "new_weight", 
                                                "time_label", "time", 
                                                "label")), \
            f"cls_token_type can only be one of {vals}."

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
            "cls_token_regularizer_ids", 
            id_less_than_key=False, 
            allowed_values=[None]+list(range(local_vars["depth"]+1))
        )
        self._check_dict_assertions(
            local_vars, 
            "cls_token_regularizer_kwargs", 
            check_items_num=False, 
            id_less_than_key=False, 
            allowed_keys=(cls_token_regularizer_kwargs:=(
                "start", "end"
            )), 
            check_values=False, 
        ); self.cls_token_regularizer_kwargs = cls_token_regularizer_kwargs

        assert local_vars["cross_attention_plug_type"] in ("values", "queries"), \
            "cross_attention_plug_type can only be values or queries."

    def _fill_none_ids(self, ids_dict: dict, 
                       min_id: int = 0, 
                       max_id: int | None = None):
        for key in ids_dict:
            max_id_ = key if max_id is None else max_id+1
            ids_dict[key] = list(range(min_id, max_id_)) \
                            if None in ids_dict[key] \
                            else ids_dict[key]

        return ids_dict

    def _fix_negative_ids(self, ids_dict: dict, 
                        depth: int):
        for key, value in ids_dict.items():
            value = list(value)

            for i, id_ in enumerate(value):
                if id_ < 0:
                    value[i] = id_ + depth + 1

            ids_dict[key] = value

        return ids_dict

    def _handle_ids(self, ids_dict: dict | list, 
                    depth: int, min_id: int = 0, 
                    max_id: int | None = None):
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

        if not_dict:
            return ids_dict[1]
        return ids_dict

    def _handle_all_ids(self):
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

    def _get_layers_dict_last_output_dim(self, layers_dict: dict, 
                                        skip_reshaper: bool):
        last_output_dim = None

        if (key:=self.FC) in layers_dict:
            last_output_dim = layers_dict[key].output_dim
        if (key:=self.VTB) in layers_dict:
            last_output_dim = layers_dict[key].output_dim
        if (key:=self.LM) in layers_dict:
            last_output_dim = layers_dict[key].output_dim
        if (key:=self.DS) in layers_dict:
            last_output_dim = layers_dict[key].output_dim
        if (key:=self.US) in layers_dict:
            last_output_dim = layers_dict[key].output_dim
        if (key:=self.R) in layers_dict and not skip_reshaper:
            last_output_dim = layers_dict[key].target_shape[-1]

        return last_output_dim

    def _get_last_output_dim(self, i: int, 
                            layers_dicts: list[dict], 
                            base_dim: int, 
                            skip_reshaper: bool = False):
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

    def _get_current_output_dim(self, i: int, 
                                layers_dicts: list[dict], 
                                layers_dict: dict, 
                                base_dim: int, 
                                skip_reshaper: bool = False):
        layers_dicts = layers_dicts + [layers_dict]
        output_dim = self._get_last_output_dim(
            i, 
            layers_dicts, 
            base_dim, 
            skip_reshaper=skip_reshaper
        )

        return output_dim

    def _get_unforced_total_dim(self, ids_set: list[int], 
                                layers_dicts: list[dict], 
                                base_dim: int, 
                                skip_reshaper: bool = False, 
                                kwargs: dict | None = None):
        dims = []

        for i in ids_set:
            if i == 0:
                dims.append(base_dim)
            else:
                dims.append(self._get_last_output_dim(
                    i-1, 
                    layers_dicts, 
                    base_dim, 
                    skip_reshaper=skip_reshaper
                ))

        if kwargs is None or \
            kwargs.get("connect_type", "concat") == "concat":
            return sum(dims)

        for dim_1 in dims:
            for dim_2 in dims:
                assert dim_1 == dim_2, \
                    "In connect_type == add, all of the feature dimensions must be equal."

        return dims[0]

    def _get_last_grid_size(self, i: int, 
                            layers_dicts: list[dict], 
                            base_grid_size: int, 
                            skip_reshaper: bool = False):
        if i == -1:
            return base_grid_size

        grid_size = None
        if i < len(layers_dicts):
            if (key:=self.LM) in layers_dicts[i]:
                grid_size = layers_dicts[i][key].output_grid_size
            if (key:=self.DS) in layers_dicts[i]:
                grid_size = layers_dicts[i][key].output_grid_size
            if (key:=self.US) in layers_dicts[i]:
                grid_size = layers_dicts[i][key].output_grid_size
            if (key:=self.R) in layers_dicts[i] and not skip_reshaper:
                grid_size = target_shape[1] \
                            if len(target_shape:=layers_dicts[i][key].target_shape) == 3 else \
                            None

        grid_size = self._get_last_grid_size(
            i-1, 
            layers_dicts, 
            base_grid_size
        ) if grid_size is None else grid_size

        return grid_size

    def _get_current_grid_size(self, i: int, 
                            layers_dicts: list[dict], 
                            layers_dict: dict, 
                            base_grid_size: int, 
                            skip_reshaper: bool = False):
        layers_dicts = layers_dicts + [layers_dict]
        grid_size = self._get_last_grid_size(
            i, 
            layers_dicts, 
            base_grid_size, 
            skip_reshaper=skip_reshaper
        )

        return grid_size

    def _get_ids_grid_size(self, ids_set: list[int], 
                        layers_dicts: list[dict], 
                        base_grid_size: int, 
                        must_be_same: bool = False):
        grid_sizes = []

        for i in ids_set:
            if i == 0:
                grid_sizes.append(base_grid_size)
            else:
                grid_sizes.append(self._get_last_grid_size(
                    i=i-1, 
                    layers_dicts=layers_dicts, 
                    base_grid_size=base_grid_size
                ))

        if not must_be_same:
            return grid_sizes

        for grid_size_1 in grid_sizes:
            for grid_size_2 in grid_sizes:
                assert grid_size_1 == grid_size_2, \
                    "All of the feature grid sizes must be equal."

        return grid_sizes[0]

    def _create_time_embedder(self, name_prefix: str = ""):
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

    def _create_label_embedder(self, name_prefix: str = ""):
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

    def _create_merger(self, merger_type: MergeType, 
                    name: str | None = None):
        if merger_type == "concat":
            merger_layer = layers.Concatenate(
                axis=-1, 
                name=name
            )
        elif merger_type == "add":
            merger_layer = layers.Add(
                name=name
            )
        else:
            raise ValueError("conds_merger_type can be either concat or add.")

        return merger_layer

    def _create_embedders(self):
        self._cond_type = self.cond_type if self.cond_type is not None and \
                        (not self.ln_no_adaptation or self.patches_conds_merger_type is not None) \
                        else []
        self._cls_token_type = self.cls_token_type if self.cls_token_type is not None else []

        embed_times_flag = "time" in self._cls_token_type or "time" in self._cond_type
        embed_labels_flag = "label" in self._cls_token_type or "label" in self._cond_type # or 0 in self.cls_token_regularizer_ids
        conds_merger_type_flag = ("time" in self._cls_token_type and "label" in self._cls_token_type) or (
                                "time" in self._cond_type and "label" in self._cond_type)

        self.patch_embedder = PatchEmbedding(
            dim=self.patches_dim, 
            grid_size=self.grid_size, 
            pos_embed_type=self.patches_pos_embed_type, 
            pos_merger_type=self.patches_pos_merger_type, 
            patch_size=self.patch_size, 
            patchify_with_cnn=self.patchify_with_cnn, 
            shift_right_token=self.shift_inputs, 
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
            name=f"{self.name_prefix}depth_0_{self.CTR[2:]}"
        ) if 0 in self.cls_token_regularizer_ids else None

    def _create_cls_token(self, dim: int, 
                        cls_token_pos_merger_type: MergeType, 
                        cls_token_freq_dim: int, 
                        cls_token_mlp_ratio: float, 
                        cls_token_type: TokenType, 
                        name_prefix: str = ""):
        cls_token = SingleTokenLayer(
            dim=dim, 
            pos_merger_type=cls_token_pos_merger_type, 
            embed_freq_dim=cls_token_freq_dim, 
            mlp_ratio=cls_token_mlp_ratio, 
            input_as_token=cls_token_type in ("time_label", "time", "label"), 
            name=f"{self.name_prefix}{name_prefix}depth_0_cls_token"
        )

        return cls_token

    def _create_feature_handler(self, ids_set: list[int], 
                                layers_dicts: list[dict], 
                                base_dim: int, dim_forced: bool, 
                                ln_mlp_ratio: float, ln_no_adaptation: bool, 
                                kwargs: dict, zero_index_base_dim: int | None = None, 
                                increased_dim: int = 0, output_dim_flag: bool = True, 
                                name: str | None = None):
        increased_dim_ = self._get_unforced_total_dim(
            ids_set, 
            layers_dicts, 
            base_dim=base_dim if zero_index_base_dim is None \
                    else zero_index_base_dim, 
            kwargs=kwargs
        )
        if kwargs.get("connect_type", "concat") == "concat":
            increased_dim_ += increased_dim
        elif increased_dim != 0:
            assert increased_dim_ == increased_dim, \
                "In connect_type == add, all of the feature dimensions must be equal."

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

    def _create_vit_block(self, i: int, layers_dicts: list[dict],
                        layers_dict: dict, base_dim: int, 
                        mha_key_dim: int | None, mha_value_dim: int | None, 
                        mha_num_heads: int, mlp_ratio: float, 
                        mlp_output_dim: int, ln_mlp_ratio: float, 
                        ln_no_adaptation: bool, drop_prob: float, 
                        drop_per_sample: bool, use_decoder: bool, 
                        name_prefix: str, mha_query_dim: int | None = None):
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
            "name": name_prefix
        }

        if use_decoder:
            block_kwargs["name"] += "decoder_block"
            block = DiTDecoderBlock(**block_kwargs)
        else:
            block_kwargs["name"] += "encoder_block"
            block = VisionTransformerBlock(**block_kwargs)
            
        return block

    def _create_local_mixer(self, i: int, 
                            dim_forced: bool, 
                            layers_dicts: list[dict], 
                            layers_dict: dict, 
                            base_dim: int, 
                            base_grid_size: int, 
                            ln_mlp_ratio: float, 
                            ln_no_adaptation: bool, 
                            circumvent_cls_token: bool, 
                            kwargs: dict = {}, 
                            name: str | None = None):
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
            "circumvent_cls_token": circumvent_cls_token, 
            "name": name
        }
        local_mixer_kwargs.update(kwargs)

        flag1 = kwargs.get("pos_merger_type", "add") == "concat" and kwargs.get("pos_embed_type", None)
        flag2 = kwargs.get("depth_multiplier", 1) > 1 and not kwargs.get("use_pointwise", True)
        if dim_forced and (flag1 or flag2):
            local_mixer_kwargs["mlp_output_dim"] = local_mixer_kwargs["dim"]

        cnn = LocalMixer(**local_mixer_kwargs)

        return cnn

    def _create_scaler(self, scaler_type: str, 
                       i: int, dim_forced: bool, 
                       layers_dicts: list[dict], 
                       layers_dict: dict, 
                       base_dim: int, 
                       base_grid_size: int, 
                       ln_mlp_ratio: float, 
                       ln_no_adaptation: bool, 
                       circumvent_cls_token: bool, 
                       kwargs: dict = {}, 
                       name: str | None = None):
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
            "circumvent_cls_token": circumvent_cls_token, 
            "name": name
        }
        scaler_kwargs.update(kwargs)

        flag1 = kwargs.get("pos_merger_type", "add") == "concat" and kwargs.get("pos_embed_type", None)
        flag2 = kwargs.get("cnn_dim_ratio", 1) > 1 and \
                (kwargs.get("scaling_method", "avg_pooling") in ("cnn_stride") or 
                kwargs.get("scaling_method", "cnn_transpose") in ("cnn_transpose", "cnn_interpolate"))
        if dim_forced and (flag1 or flag2):
            scaler_kwargs["mlp_output_dim"] = scaler_kwargs["dim"]

        if scaler_type == "downsample":
            scaler = Downsample(**scaler_kwargs)
        elif scaler_type == "upsample":
            scaler = Upsample(**scaler_kwargs)
        else:
            raise ValueError("scaler_type can either be downsample or upsample.")

        return scaler

    def _create_reshaper(self, reshape_type: str, 
                        i: int, layers_dicts: list[dict], 
                        layers_dict: dict, 
                        base_dim: int, 
                        base_grid_size: int, 
                        name: str | None = None):
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

        if reshape_type == "flatten":
            shape = (
                grid_size * grid_size * dim, 
            )
        elif reshape_type == "unflatten":
            shape = (
                grid_size * grid_size, 
                dim
            )
        else:
            raise ValueError("reshape_type needs to be either flatten or unflatten.")

        reshaper = layers.Reshape(
            shape, 
            name=name
        )

        return reshaper

    def _create_token_regularizer(self, name: str | None = None):
        token_regularizer = layers.Dense(
            self.num_classes, 
            activation="softmax", 
            name=name
        )

        return token_regularizer

    def _create_layers(self):
        self.layers_dicts = []

        for i in range(self.depth):
            layers_dict = {}
            key = i+1

            if key in self.connection_ids_dict:
                layers_dict[self.FC] = self._create_feature_handler(
                    ids_set=self.connection_ids_dict[key], 
                    layers_dicts=self.layers_dicts, 
                    base_dim=self.dim, 
                    dim_forced=self.dim_forced, 
                    ln_mlp_ratio=self.ln_mlp_ratio, 
                    ln_no_adaptation=self.ln_no_adaptation, 
                    kwargs=self.connection_kwargs, 
                    name=f"{self.name_prefix}depth_{key}_{self.FC[2:]}"
                )
            
            if key in self.cross_attention_ids_dict:
                layers_dict[self.CAC] = self._create_feature_handler(
                    ids_set=self.cross_attention_ids_dict[key], 
                    layers_dicts=self.layers_dicts, 
                    base_dim=self.dim, 
                    dim_forced=self.dim_forced, 
                    ln_mlp_ratio=self.ln_mlp_ratio, 
                    ln_no_adaptation=self.ln_no_adaptation, 
                    kwargs=self.cross_attention_kwargs, 
                    name=f"{self.name_prefix}depth_{key}_{self.CAC[2:]}"
                )

            if key in self.vit_block_ids:
                mha_query_dim = layers_dict[self.CAC].output_dim if \
                                key in self.cross_attention_ids_dict and \
                                self.cross_attention_plug_type == "queries" \
                                else None

                layers_dict[self.VTB] = self._create_vit_block(
                    i=i, layers_dicts=self.layers_dicts, 
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

            if key in self.local_mixer_ids:
                layers_dict[self.LM] = self._create_local_mixer(
                    i=i, 
                    dim_forced=self.dim_forced, 
                    layers_dicts=self.layers_dicts, 
                    layers_dict=layers_dict, 
                    base_dim=self.dim, 
                    base_grid_size=self.grid_size, 
                    ln_mlp_ratio=self.ln_mlp_ratio, 
                    ln_no_adaptation=self.ln_no_adaptation, 
                    circumvent_cls_token=self.cls_token_type is not None, 
                    kwargs=self.local_mixer_kwargs, 
                    name=f"{self.name_prefix}depth_{key}_{self.LM[2:]}"
                )

            if key in self.downsample_ids:
                layers_dict[self.DS] = self._create_scaler(
                    scaler_type="downsample", 
                    i=i, 
                    dim_forced=self.dim_forced, 
                    layers_dicts=self.layers_dicts, 
                    layers_dict=layers_dict, 
                    base_dim=self.dim, 
                    base_grid_size=self.grid_size, 
                    ln_mlp_ratio=self.ln_mlp_ratio, 
                    ln_no_adaptation=self.ln_no_adaptation, 
                    circumvent_cls_token=self.cls_token_type is not None, 
                    kwargs=self.downsample_kwargs, 
                    name=f"{self.name_prefix}depth_{key}_{self.DS[2:]}"
                )

            if key in self.upsample_ids:
                layers_dict[self.US] = self._create_scaler(
                    scaler_type="upsample", 
                    i=i, 
                    dim_forced=self.dim_forced, 
                    layers_dicts=self.layers_dicts, 
                    layers_dict=layers_dict, 
                    base_dim=self.dim, 
                    base_grid_size=self.grid_size, 
                    ln_mlp_ratio=self.ln_mlp_ratio, 
                    ln_no_adaptation=self.ln_no_adaptation, 
                    circumvent_cls_token=self.cls_token_type is not None, 
                    kwargs=self.upsample_kwargs, 
                    name=f"{self.name_prefix}depth_{key}_{self.US[2:]}"
                )

            if key in self.reshaper_ids_dict:
                layers_dict[self.R] = self._create_reshaper(
                    reshape_type=self.reshaper_ids_dict[key], 
                    i=i, layers_dicts=self.layers_dicts, 
                    layers_dict=layers_dict, base_dim=self.dim, 
                    base_grid_size=self.grid_size, 
                    name=f"{self.name_prefix}depth_{key}_{self.R[2:]}"
                )

            if key in self.cls_token_regularizer_ids:
                layers_dict[self.CTR] = self._create_token_regularizer(
                    name=f"{self.name_prefix}depth_{key}_{self.CTR[2:]}"
                )

            self.layers_dicts.append(layers_dict)

    def get_variables_names(self, vars: list[tf.Variable] | None = None):
        vars = self.trainable_variables if vars is None else vars
        names = [var.name for var in vars]

        return names

    def build_model(self, call_model: bool = True):
        noisy_images = layers.Input(
            shape=(
                self.image_size, 
                self.image_size, 
                self.channels
            ),
            dtype=tf.float32, 
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
        self.outputs = self.call(self.inputs) if call_model else None

        input_shape = [
            input_layer.shape for input_layer in self.inputs
        ]

        return input_shape

    def build(
        self, 
        input_shape: tuple[tuple, tuple, tuple] | None = None
    ):
        input_shape = self.build_model()
        super().build(input_shape)

    def embed_conditions(self, times: tf.Tensor, 
                        labels: tf.Tensor, 
                        cond_type: CondType, 
                        full_return: bool = False, 
                        training: bool | None = None):
        cond_type = [] if cond_type is None else cond_type

        time_embeds = self.time_embedder(
            times, 
            training=training
        ) if self.time_embedder is not None and "time" in cond_type else None

        label_embeds = self.label_embedder(
            labels, 
            training=training
        ) if self.label_embedder is not None and "label" in cond_type else None

        conds = self.conds_merger(
            (time_embeds, label_embeds), 
            training=training
        ) if self.time_embedder is not None and self.label_embedder is not None \
            and "time" in cond_type and "label" in cond_type else None

        if conds is None:
            conds = time_embeds if time_embeds is not None else label_embeds

        if full_return:
            return conds, time_embeds, label_embeds
        return conds

    def embed_inputs(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        cond_type: CondType, 
        full_return: bool = False, 
        training: bool | None = None
    ):
        images, times, labels = inputs

        conds_list = self.embed_conditions(
            times, labels, cond_type, 
            full_return=full_return, 
            training=training
        )
        conds_merged = conds_list[0] if full_return else conds_list

        x = self.patch_embedder(
            images, 
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

    def prepend_cls_token(self, x: tf.Tensor, 
                        cls_token_type: TokenType, 
                        time_embeds: tf.Tensor | None = None, 
                        label_embeds: tf.Tensor | None = None, 
                        times: tf.Tensor | None = None, 
                        labels: tf.Tensor | None = None, 
                        training: bool | None = None):
        if cls_token_type == "time":
            embeds = self.time_embedder(
                times, 
                training=training
            ) if time_embeds is None else time_embeds
        elif cls_token_type == "label":
            embeds = self.label_embedder(
                labels, 
                training=training
            ) if label_embeds is None else label_embeds
        elif cls_token_type == "time_label":
            embeds = self.embed_conditions(
                times, labels, 
                cls_token_type, 
                full_return=False, 
                training=training
            )
        elif cls_token_type == "new_weight":
            embeds = None

        x = tf.concat([
            self.cls_token(
                (x, embeds), 
                training=training
            ), 
            x
        ], axis=1)

        return x

    def slice_and_flatten_tokens(self, x, end, start):
        x = tf.reshape(
            x[:, start: end, :], 
            (
                -1, 
                tf.shape(x)[-1] * (end-start)
            )
        )

        return x

    def refine_with_cnn(self, x: tf.Tensor, 
                        training: bool | None = None):
        h = self.final_cnn(
            x, 
            training=training
        )
        x = x + h if self.final_cnn_residual else h 

        return x

    def unpatchify(self, x: tf.Tensor, 
                cond: tf.Tensor, 
                training: bool | None = None):
        x = self.final_layer_norm(
            (x, cond), 
            training=training
        )
        x = self.final_ffn(
            x, 
            training=training
        )

        x = tf.reshape(x, (
            -1, 
            self.grid_size, 
            self.grid_size, 
            self.patch_size, 
            self.patch_size, 
            self.channels
        ))
        x = tf.transpose(x, 
            (0, 1, 3, 2, 4, 5)
        )
        x = tf.reshape(x, (
            -1, 
            self.image_size, 
            self.image_size, 
            self.channels
        ))

        x = self.refine_with_cnn(
            x, 
            training=training
        ) if self.use_final_cnn else x
        x = self.final_activation(
            x, 
            training=training
        )

        return x

    def encode(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        max_depth: int = -1, 
        training: bool | None = None
    ):
        x, (cond, time_embeds, label_embeds) = self.embed_inputs(
            inputs, 
            self.cond_type, 
            full_return=True, 
            training=training
        )
        x = self.prepend_cls_token(
            x, self.cls_token_type, 
            time_embeds=time_embeds, 
            label_embeds=label_embeds, 
            times=inputs[1], 
            labels=inputs[2], 
            training=training
        ) if self.cls_token_type is not None else x

        z = self.labels_embed_reg(
            label_embeds, 
            training=training
        ) if self.labels_embed_reg is not None else None

        features_list = [x]
        regs_list = [z]
        for i, layers_dict in enumerate(self.layers_dicts):
            if i == max_depth:
                break

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

            x = layers_dict[self.R](
                x, 
                training=training
            ) if self.R in layers_dict else x

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

        x = x[:, 1:] if self.cls_token_type is not None else x

        return x, cond, features_list, regs_list

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        full_return: bool = False, 
        training: bool | None = None
    ):
        x, cond, features_list, regs_list = self.encode(
            inputs, 
            training=training
        )
        noises = self.unpatchify(
            x, cond, 
            training=training
        ) if self.use_unpatchify else x

        if full_return:
            return noises, cond, features_list, regs_list
        return noises
