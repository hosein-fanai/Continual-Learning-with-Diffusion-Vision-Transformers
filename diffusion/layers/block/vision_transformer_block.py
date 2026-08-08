from tensorflow.keras import layers

from diffusion.layers.base_layer import BaseLayer
from diffusion.layers.drop_path import DropPath


class VisionTransformerBlock(BaseLayer):
    """
    
    """

    def __init__(
        self, 
        mlp_ratio: int | None = 4, 
        mlp_activation_func: str = "gelu", 
        dim: int = 32, 
        key_dim: int | None = None, 
        value_dim: int | None = None, 
        query_dim: int | None = None, 
        num_heads: int = 4, 
        gate_query_flag: bool = True, 
        drop_prob: float = 0., 
        drop_per_sample: bool = True, 
        **kwargs
    ):
        super().__init__(
            use_layer_norm=True, 
            ln_dim=dim, 
            mlp_ratio=mlp_ratio, 
            mlp_activation_func=mlp_activation_func, 
            **kwargs
        )
        self._save_init_args(locals())

        self.key_dim = self.dim // self.num_heads if self.key_dim is None else self.key_dim
        self.mlp_output_dim = self.dim if self.mlp_output_dim is None else self.mlp_output_dim
        self.query_dim = self.dim if self.query_dim is None else self.query_dim

        self.mha_layer_norm = self._create_layer_norm(
            gate_dim=self.query_dim if self.gate_query_flag else self.dim, 
            name=f"{self.name}/mha_layer_norm"
        )
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.key_dim, 
            value_dim=self.value_dim, 
            name="mha"
        )
        self.mha_residual_projector = layers.Dense(
            self.query_dim, 
            name="mha_residual_projector"
        ) if self.query_dim != self.dim else None
        self.mha_drop_path = DropPath(
            drop_prob=self.drop_prob, 
            per_sample=self.drop_per_sample, 
            name=f"{self.name}/mha_drop_path"
        )

        self.mlp_layer_norm = self._create_layer_norm(
            dim=self.query_dim, 
            gate_dim=self.mlp_output_dim, 
            name=f"{self.name}/mlp_layer_norm"
        )
        self.mlp = self._create_mlp(
            self.query_dim
        )
        self.mlp_residual_projector = layers.Dense(
            self.mlp_output_dim, 
            name="mlp_residual_projector"
        ) if self.mlp_output_dim != self.query_dim else None
        self.mlp_drop_path = DropPath(
            drop_prob=self.drop_prob, 
            per_sample=self.drop_per_sample, 
            name=f"{self.name}/mlp_drop_path"
        )

    def _call_self_attention(
        self, 
        x, 
        cond, 
        queries, 
        values, 
        mask, 
        training
    ): # also can perform cross attention
        h, gate = self.mha_layer_norm(
            (x, cond), 
            training=training
        )
        h = self.mha(
            query=h if queries is None else queries, 
            value=h if values is None else values, 
            attention_mask=mask, 
            training=training
        )
        x = self.mha_residual_projector(
            x, 
            training=training
        ) if x.shape[-1] != h.shape[-1] else x
        x = x + self.mha_drop_path(
            gate * h, 
            training=training
        )

        return x

    def _call_mlp(
        self, 
        x, 
        cond, 
        training
    ):
        h, gate = self.mlp_layer_norm(
            (x, cond), 
            training=training
        )
        h = self.mlp(
            h, 
            training=training
        )
        x = self.mlp_residual_projector(
            x, 
            training=training
        ) if x.shape[-1] != h.shape[-1] else x
        x = x + self.mlp_drop_path(
            gate * h, 
            training=training
        )

        return x

    def call(
        self, 
        inputs, 
        queries=None, 
        values=None, 
        mask=None, 
        training=None
    ):
        x, cond = inputs

        x = self._call_self_attention(
            x, cond, 
            queries, 
            values, 
            mask, 
            training
        )
        x = self._call_mlp(
            x, cond, 
            training
        )

        return x
