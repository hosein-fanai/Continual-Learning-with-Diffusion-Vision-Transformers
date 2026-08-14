import tensorflow as tf

from typing import get_args

from diffusion.layers.embedding import MergeType
from diffusion.layers.base_layer import BaseLayer


class FeatureHandler(BaseLayer):
    """
    
    """

    def __init__(
        self, 
        ids: list[int] | None = None, 
        connect_axis: int = -1, 
        connect_type: MergeType = "concat", 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._save_init_args(locals())
        self._check_fh_assertions(locals())

        self.layer_norm = self._create_layer_norm(
            return_gate=False
        )
        self.mlp = self._create_mlp(
            self.ln_dim
        )

    def _check_fh_assertions(self, local_vars):
        assert local_vars["connect_type"] in get_args(MergeType), \
            f"connect_type can be one of {get_args(MergeType)}."

        if self.mlp_output_dim is not None:
            assert self.ln_dim is not None, \
                "ln_dim cannot be None when mlp_output_dim is not None."

    def call(
        self, 
        features_list, 
        second_list=None, 
        ids=None, 
        cond=None, 
        training=None
    ):
        second_list = [] if second_list is None else second_list
        ids = self.ids if ids is None else ids

        if len(ids) == 0 and len(second_list) == 0:
            return None

        selected_features = [
            features_list[id_] for id_ in ids
        ] + list(second_list)

        if self.connect_type == "concat":
            x = tf.concat(
                selected_features, 
                axis=self.connect_axis
            )
        else:
            x = sum(selected_features)

        x = self.layer_norm(
            (x, cond), 
            training=training
        ) if self.layer_norm is not None else x
        x = self.mlp(
            x, 
            training=training
        ) if self.mlp is not None else x

        return x
