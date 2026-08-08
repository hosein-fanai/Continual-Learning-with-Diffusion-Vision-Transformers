from tensorflow.keras import layers

from diffusion.layers.block.vision_transformer_block import VisionTransformerBlock
from diffusion.layers.drop_path import DropPath


class DiTDecoderBlock(VisionTransformerBlock):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            gate_query_flag=False, 
            **kwargs
        )

        self.mha_layer_norm2 = self._create_layer_norm(
            gate_dim=self.query_dim, 
            name=f"{self.name}/mha_layer_norm_2"
        )
        self.mha2 = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.key_dim, 
            value_dim=self.value_dim, 
            name="mha_2"
        )
        self.mha_drop_path2 = DropPath(
            drop_prob=self.drop_prob, 
            per_sample=self.drop_per_sample, 
            name=f"{self.name}/mha_drop_path_2"
        )

    def _call_cross_attention(
        self, 
        x, 
        cond, 
        queries, 
        values, 
        mask, 
        training
    ):
        h, gate = self.mha_layer_norm2(
            (x, cond), 
            training=training
        )
        h = self.mha2(
            query=h if queries is None else queries, 
            value=h if values is None else values, 
            attention_mask=mask, 
            training=training
        )
        x = self.mha_residual_projector(
            x, 
            training=training
        ) if x.shape[-1] != h.shape[-1] else x
        x = x + self.mha_drop_path2(
            gate * h, 
            training=training
        )

        return x

    def call(
        self, 
        inputs, 
        queries=None, 
        values=None, 
        causal_mask=None, 
        training=None
    ):
        x, cond = inputs

        x = self._call_self_attention(
            x, cond, 
            None, None, 
            causal_mask, 
            training
        )
        x = self._call_cross_attention(
            x, cond, 
            queries, 
            values, 
            None, 
            training
        )
        x = self._call_mlp(
            x, cond, 
            training
        )

        return x
