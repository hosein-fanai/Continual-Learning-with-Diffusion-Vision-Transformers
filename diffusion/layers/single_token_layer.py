import tensorflow as tf
from tensorflow.keras import initializers

from diffusion.layers.embedding.base_embedding import BaseEmbedding


class SingleTokenLayer(BaseEmbedding):

    def __init__(
        self, 
        with_pos_embed: bool = True, 
        input_as_token: bool = False, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._save_init_args(locals())

        self.mlp_ratio = 1 if self.mlp_ratio is None and self.embed_freq_dim is not None \
                        else self.mlp_ratio
        self.mlp_output_dim = self.dim if self.mlp_output_dim is None \
                            and self.embed_freq_dim is not None \
                            else self.mlp_output_dim
        self.pos_embed_type = "new_weight" if self.with_pos_embed else None
        self.embed_dim = self.dim // 2 if self.pos_merger_type == "concat" else self.dim

        self.token = self.add_weight(
            shape=(1, 1, self.embed_dim), 
            initializer=initializers.RandomNormal(stddev=1e-6), # or "zeros"
            trainable=True, 
            name=f"{self.name}/token_embeddings"
        ) if not self.input_as_token else None
        self.token_mlp = self._create_mlp(
            self.embed_dim
        ) if self.token is not None else None
        self.pos_embed = self._create_embeddings(
            output_grid_size=1, 
        )
        self.pos_embed_mlp = self._create_mlp(
            self.embed_dim
        ) if self.pos_embed is not None else None

    def call(self, inputs, training=None):
        images, token = inputs

        x = token[:, None, :] if self.input_as_token else self.token
        x = self.token_mlp(
            x, 
            training=training
        ) if self.token_mlp is not None else x
        x = self._pos_merger(
            x, 
            batch_size=tf.shape(images)[0], 
            training=training
        )

        return x
