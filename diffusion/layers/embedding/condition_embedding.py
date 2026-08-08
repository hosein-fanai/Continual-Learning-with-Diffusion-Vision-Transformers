from diffusion.layers.embedding.base_embedding import BaseEmbedding


class ConditionEmbedding(BaseEmbedding):

    def __init__(
        self, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._save_init_args(locals())

        self.mlp_ratio = 1 if self.mlp_ratio is None and self.embed_freq_dim is not None \
                        else self.mlp_ratio
        self.mlp_output_dim = self.dim if self.mlp_output_dim is None \
                            and self.embed_freq_dim is not None \
                            else self.mlp_output_dim

        self.embed = self._create_embedding_layer()
        self.embed_mlp = self._create_mlp(
            self.embed_dim
        )

    def call(self, x, training=None):
        x = self.embed(
            x, 
            training=training
        )
        x = self.embed_mlp(
            x, 
            training=training
        ) if self.embed_mlp is not None else x

        return x
