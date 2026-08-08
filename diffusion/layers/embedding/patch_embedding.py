import tensorflow as tf
from tensorflow.keras import layers, models

from diffusion.layers.embedding.base_embedding import BaseEmbedding
from diffusion.layers.single_token_layer import SingleTokenLayer


class PatchEmbedding(BaseEmbedding):
    """
    """

    def __init__(
        self, 
        patch_size: int = 2, 
        patchify_with_cnn: bool = False, 
        shift_right_token: bool = False, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._save_init_args(locals())

        self.mlp_ratio = 1 if self.mlp_ratio is None and self.embed_freq_dim is not None \
                        else self.mlp_ratio
        self.hidden_dim = self.dim // 2 if self.pos_merger_type == "concat" \
                        else self.dim
        self.mlp_output_dim = self.hidden_dim if self.mlp_output_dim is None \
                            and self.embed_freq_dim is not None \
                            else self.mlp_output_dim
        self.embed_dim = self.hidden_dim if self.embed_freq_dim is None \
                        else self.embed_freq_dim

        if self.patchify_with_cnn:
            self.patch_projector = models.Sequential([
                layers.Conv2D(
                    self.hidden_dim // 2, 
                    kernel_size=3, 
                    strides=1, 
                    padding="same", 
                    activation="swish", 
                    name=f"{self.name}/patch_projector/conv_1"
                ), 
                layers.Conv2D(
                    self.hidden_dim, 
                    kernel_size=3, 
                    strides=self.patch_size, 
                    padding="same", 
                    name=f"{self.name}/patch_projector/conv_2"
                )
            ], name="patch_projector")
        else:
            self.patch_projector = layers.Conv2D(
                self.hidden_dim, 
                self.patch_size, 
                strides=self.patch_size, 
                name="patch_projector"
            )

        self.shift_right_token = SingleTokenLayer(
            dim=self.hidden_dim, 
            with_pos_embed=False, 
            name=f"{self.name}/bos_token"
        ) if self.shift_right_token else None
        self.pos_embed = self._create_embeddings(
            output_grid_size=self.grid_size, 
        ) if self.pos_merger_type is not None else None
        self.pos_embed_mlp = self._create_mlp(
            self.embed_dim
        )

    def call(self, x, training=None):
        B = tf.shape(x)[0]

        x = self.patch_projector(
            x, 
            training=training
        )
        x = tf.reshape(x, (
            B, 
            self.grid_size * self.grid_size, 
            -1
        ))
        x = tf.concat([
            self.shift_right_token(
                (x, None), 
                training=training
            ), 
            x[:, :-1, :]
        ], axis=1) if self.shift_right_token is not None else x
        x = self._pos_merger(
            x, 
            training=training
        )

        return x
