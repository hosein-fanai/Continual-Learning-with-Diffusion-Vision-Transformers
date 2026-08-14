import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

from typing import get_args

from . import PosEmbedType

from diffusion.layers.embedding import MergeType
from diffusion.layers.base_layer import BaseLayer


class BaseEmbedding(BaseLayer):
    """
    
    """

    def __init__(
        self, 
        dim: int, 
        grid_size: int | None = None, 
        pos_embed_type: PosEmbedType | None  = "new_weight", 
        pos_interpolation_method: str = "bicubic", 
        pos_merger_type: MergeType = "add", 
        embed_freq_dim: int | None = None, 
        embed_steps: int | None = None, 
        embed_temperature: float = 10_000., 
        embed_trainable: bool = False, 
        **kwargs
    ):
        super().__init__(
            ln_dim=dim, 
            **kwargs
        )
        self._save_init_args(locals())

        assert self.pos_embed_type in (None,)+get_args(PosEmbedType), \
        f"pos_embed_type needs to be None or one of {get_args(PosEmbedType)}."
        assert self.pos_merger_type in get_args(MergeType), \
            f"pos_merger needs to be one of {get_args(MergeType)}."


        self.embed_dim = self.dim if self.embed_freq_dim is None else self.embed_freq_dim
        self.pos_embed_mlp = None

    def _get_1d_sincos_embedding(self, dim: int, 
                                positions: np.ndarray, 
                                temperature: float = 10_000.
                                ) -> np.ndarray:
        """Return a sinusoidal embedding with exactly ``dim`` channels.
        """

        frequency_count = max((dim + 1) // 2, 1)
        frequencies = 1. / (temperature ** (
            np.arange(frequency_count, dtype=np.float32) /
            float(frequency_count)
        ))
        angles = positions[:, None] * frequencies[None, :]

        embedding = np.concatenate([
            np.sin(angles), 
            np.cos(angles)
        ], axis=-1)
        embedding = embedding[:, :dim]

        return embedding

    def _get_2d_sincos_embedding(self, dim: int, 
                                grid_size: int, 
                                temperature: float = 10_000., 
                                name: str | None = None
                                ) -> tf.Tensor:
        """Build a fixed row-major 2D sine/cosine positional table.

        Unlike the common implementation that requires ``dim % 4 == 0``, this
        function supports every positive channel dimension by splitting the
        channels between the horizontal and vertical coordinates and truncating
        the final sine/cosine pair when necessary.
        """

        grid_y, grid_x = np.meshgrid(
            np.arange(grid_size, dtype=np.float32), 
            np.arange(grid_size, dtype=np.float32), 
            indexing="ij"
        )
        x_positions = grid_x.reshape(-1)
        y_positions = grid_y.reshape(-1)

        x_dim = dim // 2
        y_dim = dim - x_dim

        embedding = np.concatenate([
            self._get_1d_sincos_embedding(
                x_dim, x_positions, temperature
            ),
            self._get_1d_sincos_embedding(
                y_dim, y_positions, temperature
            ),
        ], axis=-1)
        embedding = tf.convert_to_tensor(
            embedding[None, ...], 
            dtype=tf.float32, 
            name=name
        )

        return embedding

    def _get_t_embedding(self, t, dim): # old
        half = dim // 2
        freqs = np.exp(-np.log(10_000) * np.arange(half) / half)
        args = t[:, None] * freqs[None]

        emb = tf.concat([
            tf.sin(args), 
            tf.cos(args)
        ], axis=-1)

        return emb

    def _get_2d_pos_embed(self, h, w, dim): # old
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w))
        grid = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

        emb = []
        for i in range(dim // 4):
            freq = 1.0 / (10_000 ** (i / (dim//4)))
            emb.append(np.sin(grid * freq))
            emb.append(np.cos(grid * freq))

        emb = np.concatenate(emb, axis=-1)
        emb = tf.convert_to_tensor(emb[None], dtype=tf.float32)

        return emb

    def _create_embeddings(
        self, 
        embed_dim: int | None = None, 
        grid_size: int | None = None, 
        output_grid_size: int | None = None, 
        pos_embed_type: PosEmbedType | None = None, 
        embed_steps: int | None = None, 
        temperature: float | None = None, 
        name: str | None = None
    ):
        """Create one of the supported positional representations.

        ``new_weight``
            A learned table is created directly at the target resolution.

        ``1d_interpolate`` / ``2d_interpolate``
            Creates a fixed sine/cosine table at the source resolution and
            resizes it to the target resolution.

        ``1d_learned_interpolate`` / ``2d_learned_interpolate``
            Creates a learned table at the source resolution and resizes it to
            the target resolution.

        ``1d_sincos`` / ``2d_sincos``
            Uses a fixed, parameter-free sine/cosine table at the target
            resolution.
        """

        embed_dim = self.embed_dim if embed_dim is None else embed_dim
        grid_size = self.grid_size if grid_size is None else grid_size
        pos_embed_type = None if pos_embed_type is None and self.pos_embed_type is None \
                        else pos_embed_type or self.pos_embed_type
        embed_steps = self.embed_steps if embed_steps is None else embed_steps
        temperature = self.embed_temperature if temperature is None else temperature
        name = f"{self.name}/positional_embeddings" if name is None else name

        self.pos_embed_type = pos_embed_type

        if pos_embed_type is None:
            return None

        if pos_embed_type == "new_weight":
            return self.add_weight(
                shape=(
                    1, 
                    output_grid_size * output_grid_size, 
                    embed_dim
                ), 
                initializer="zeros", 
                trainable=True, 
                name=name
            )

        if pos_embed_type == "1d_sincos":
            spatial_embedding = embed_steps is None
            embedding = self._get_1d_sincos_embedding(
                dim=embed_dim, 
                positions=tf.range(
                    0, 
                    output_grid_size * output_grid_size 
                    if spatial_embedding else embed_steps, 
                    1, 
                    dtype=tf.float32
                ),
                temperature=temperature
            )

            return embedding[None, ...] if spatial_embedding else embedding

        if pos_embed_type == "1d_interpolate":
            return self._get_1d_sincos_embedding(
                dim=embed_dim,
                positions=tf.range(
                    0, grid_size * grid_size, 1, dtype=tf.float32
                ),
                temperature=temperature
            )[None, ...]

        if pos_embed_type == "1d_learned_interpolate":
            return self.add_weight(
                shape=(
                    1,
                    grid_size * grid_size,
                    embed_dim
                ),
                initializer="zeros",
                trainable=True,
                name=name
            )

        if pos_embed_type == "2d_sincos":
            return self._get_2d_sincos_embedding(
                dim=embed_dim, grid_size=output_grid_size, 
                temperature=temperature, name=name
            )

        if pos_embed_type == "2d_interpolate":
            return self._get_2d_sincos_embedding(
                dim=embed_dim, grid_size=grid_size, 
                temperature=temperature, name=name
            )

        if pos_embed_type == "2d_learned_interpolate":
            return self.add_weight(
                shape=(
                    1, 
                    grid_size, 
                    grid_size, 
                    embed_dim
                ), 
                initializer="zeros", 
                trainable=True, 
                name=name
            )

    def _create_embedding_layer(self, **kwargs):
        kwargs["pos_embed_type"] = None if kwargs.get("pos_embed_type", None) is None \
                                and self.pos_embed_type is None \
                                else kwargs.get("pos_embed_type", None) or self.pos_embed_type
        embedding_layer = layers.Embedding(
            input_dim=kwargs.get("embed_steps", self.embed_steps), 
            output_dim=kwargs.get("embed_dim", self.embed_dim), 
            trainable=kwargs.get("embed_trainable", self.embed_trainable), 
            name="embeddings"
        )

        if kwargs["pos_embed_type"] != "new_weight":
            embedding_layer.build(())
            embedding_layer.set_weights([self._create_embeddings(**kwargs)])
        else:
            embedding_layer.trainable = True

        return embedding_layer

    def _pos_merger(self, x: tf.Tensor, 
                    batch_size: int | None = None, 
                    output_grid_size: int | None = None, 
                    training: bool | None = None) -> tf.Tensor:
        """Resolve, batch-broadcast, and merge a spatial positional embedding.
        """

        if self.pos_embed_type is None:
            return x

        batch_size = tf.shape(x)[0] if batch_size is None else batch_size
        pos_embed = self.pos_embed

        interpolated_pos_embed = "interpolate" in self.pos_embed_type
        if output_grid_size is not None or interpolated_pos_embed:
            output_grid_size = self.output_grid_size if output_grid_size is None \
                            else output_grid_size
            source_grid_size = self.grid_size if interpolated_pos_embed \
                             else self.output_grid_size

            pos_embed_dim = pos_embed.shape[-1]
            if self.pos_embed_type.startswith("1d_"):
                pos_embed = tf.reshape(pos_embed, (
                    1, 
                    source_grid_size * source_grid_size, 
                    1, 
                    pos_embed_dim
                ))
                resize_shape = (
                    output_grid_size * output_grid_size, 
                    1
                )
            else:
                pos_embed = tf.reshape(pos_embed, (
                    1, 
                    source_grid_size, 
                    source_grid_size, 
                    pos_embed_dim
                ))
                resize_shape = (
                    output_grid_size, 
                    output_grid_size
                )

            pos_embed = tf.image.resize(pos_embed, 
                size=resize_shape, 
                method=self.pos_interpolation_method
            )
            pos_embed = tf.reshape(pos_embed, (
                1, 
                output_grid_size * output_grid_size, 
                pos_embed_dim
            ))
            pos_embed.set_shape((1, None, pos_embed_dim))

        pos_embed = self.pos_embed_mlp(
            pos_embed, 
            training=training
        ) if self.pos_embed_mlp is not None else pos_embed
        pos_embed = tf.repeat(
            pos_embed, 
            batch_size, 
            axis=0
        )

        if self.pos_merger_type == "concat":
            return tf.concat([x, pos_embed], axis=-1)

        if self.pos_merger_type == "add":
            return x + pos_embed
