"""Learned or input-provided single-token embeddings.

SingleTokenLayer supplies learned or input-backed class/BOS/condition tokens and
optionally merges learned positional vectors. It reuses BaseEmbedding projection
and dtype rules, broadcasts learned tokens to the runtime batch, and records
constructor settings for Keras reconstruction.
"""

import tensorflow as tf
from tensorflow.keras import initializers

from typing import Any

from common.runtime import derive_seed

from diffusion.layers.embedding.base_embedding import BaseEmbedding


class SingleTokenLayer(BaseEmbedding):
    """Produce one token for a class marker, BOS marker, or condition.

    The token is either a trainable weight or a per-example vector supplied at
    call time. An optional learned positional vector is merged by addition or
    concatenation. With concatenation, ``dim // 2`` channels are allocated to
    each half, so an even ``dim`` is required to retain exactly ``dim`` output
    channels.

    Args:
        with_pos_embed (bool): If true, force a learned ``"new_weight"`` positional
            vector and batch-broadcast the trainable token through the merge.
            If false, disable positional embeddings.
            Defaults to ``True``.
        input_as_token (bool): If true, use the second call input shaped
            ``[batch, token_dim]``. If false, ignore that input and use one
            trainable token shaped ``[1, 1, embed_dim]``.
            Defaults to ``False``.
        **kwargs (Any): :class:`BaseEmbedding` arguments. Required ``dim`` sets the
            target width. Common keys are ``pos_merger_type``,
            ``embed_freq_dim``, ``mlp_ratio``, ``mlp_output_dim``, ``name``,
            and ``dtype``. ``pos_embed_type`` is overridden by
            ``with_pos_embed``.
        seed (int | None): Optional component seed for the learned token's
            random-normal initializer.
            Defaults to ``None``.

    Inputs:
        Pair ``(images, token)``. ``images`` may be any tensor with batch as
        axis 0; only its batch size is read. ``token`` must be a floating
        ``[batch, token_dim]`` tensor when ``input_as_token=True`` and may be
        ``None`` otherwise.

    Outputs:
        Floating token tensor shaped ``[batch, 1, output_dim]``. A learned token
        is broadcast to the runtime batch before positional merging.

    Serialization:
        ``from_config(get_config())`` is supported; inherited normalization
        width is reconstructed from ``dim``.

    Attributes:
        token (tf.Variable | None): Learned [1, 1, embed_dim] token, absent for supplied
            tokens.
        token_mlp (tf.keras.Sequential | None): Optional projection of raw token vectors.
        pos_embed (tf.Variable | None): Learned positional token, absent when disabled.
        output_dim (int): Final token width after projection and positional merging.
    """

    def __init__(
        self, 
        with_pos_embed: bool = True, 
        input_as_token: bool = False, 
        seed: int | None = None,
        **kwargs: Any
    ) -> None:
        """Create the trainable token, positional vector, and projections.

        Args:
            with_pos_embed (bool): Whether to add or concatenate a learned
                positional token.
                Defaults to ``True``.
            input_as_token (bool): Whether the second call input supplies the
                token instead of a trainable weight.
                Defaults to ``False``.
            seed (int | None): Optional component seed for the learned token's
                random-normal initializer.
                Defaults to ``None``.
                None leaves component operation/initializer seeds unspecified; global
                TensorFlow RNG state can still affect draws.
            **kwargs (Any): Typed :class:`BaseEmbedding` and Keras options.

        Returns:
            None: No value is returned.
        """

        super().__init__(**kwargs)
        self._save_init_args(locals())
        derive_seed(self.seed, "single_token", "validation")

        # Split target channels between token and position only for concatenated positional
        # tokens.
        component_dim = self.dim // 2 if self.with_pos_embed and self.pos_merger_type == "concat" \
                        else self.dim
        # Add the default ratio-one projection only when a raw frequency width needs
        # projection and no ratio is set.
        self.mlp_ratio = 1 if self.mlp_ratio is None and self.embed_freq_dim is not None \
                        else self.mlp_ratio
        # Project raw frequency features to the target component width unless an output
        # width is explicit.
        self.mlp_output_dim = component_dim if self.mlp_output_dim is None and self.embed_freq_dim is not None \
                            else self.mlp_output_dim
        # Use a learned positional token when enabled; otherwise disable position merging.
        self.pos_embed_type = "new_weight" if self.with_pos_embed else None
        # Use the target width for embeddings unless a separate raw frequency width is
        # configured.
        self.embed_dim = component_dim if self.embed_freq_dim is None else self.embed_freq_dim
        # Keep an omitted component seed unseeded; otherwise normalize it to a Python
        # integer.
        self.seed = None if self.seed is None else int(self.seed)

        # Create trainable token content only when the caller will not supply token vectors.
        self.token = self.add_weight(
            shape=(1, 1, self.embed_dim), 
            initializer=initializers.RandomNormal(
                stddev=1e-6, 
                seed=derive_seed(
                    self.seed, 
                    "single_token", 
                    "initializer"
                ),
            ),
            trainable=True, 
            name=f"{self.name}/token_embeddings"
        ) if not self.input_as_token else None
        # Project supplied tokens to component width by default; otherwise use the
        # configured token projection.
        self.token_mlp = self._create_mlp(
            self.embed_dim, 
            mlp_output_dim=component_dim
        ) if self.input_as_token and self.mlp_output_dim is None else self._create_mlp(
            self.embed_dim
        )
        self.pos_embed = self._create_embeddings(
            output_grid_size=1
        )
        # Build a positional projection only when a positional table exists.
        self.pos_embed_mlp = self._create_mlp(
            self.embed_dim
        ) if self.pos_embed is not None else None

        # Count both content and positional channels only when a table is concatenated.
        self.output_dim *= 2 if self.pos_embed is not None and self.pos_merger_type == "concat" \
                        else 1

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor | None], 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor:
        """Resolve and return the configured single token.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor | None]): Pair ``(images, token)``
                described by the class contract.
            training (bool | tf.Tensor | None): Optional Keras training flag forwarded to token and
                positional projection MLPs.
                Defaults to ``None``. Keras resolves the surrounding call context; this flag is
                forwarded to child layers.

        Returns:
            tf.Tensor: of token embeddings. Its last dimension is ``dim``
            for the usual additive setup and, for concatenation, twice
            ``dim // 2`` unless an MLP overrides the component widths.
        """

        images, token = inputs

        # Add a token axis to supplied vectors, or repeat the learned token across the image
        # batch.
        x = token[:, None, :] if self.input_as_token else tf.repeat(
            self.token,
            tf.shape(images)[0],
            axis=0,
        )
        # Project token content when configured; otherwise retain the resolved token values.
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


def run_self_tests() -> dict[str, str]:
    """Test trainable and input-backed :class:`SingleTokenLayer` variants.

    Args:
        None.

    Returns:
        dict[str, str]: A one-entry mapping after positional, merge, projection, shape,
        gradient, invalid-mode, and documented serialization checks pass.
    """

    import numpy as np


    tf.random.set_seed(123)
    images = tf.zeros((3, 2, 2, 1), dtype=tf.float32)

    learned_with_position = SingleTokenLayer(dim=4, with_pos_embed=True)
    positioned = learned_with_position((images, None), training=False)
    assert positioned.shape == (3, 1, 4)
    assert learned_with_position.token.shape == (1, 1, 4)
    assert learned_with_position.pos_embed.shape == (1, 1, 4)
    assert len(learned_with_position.trainable_variables) == 2

    learned_without_position = SingleTokenLayer(dim=4, with_pos_embed=False)
    unbatched = learned_without_position((images, None), training=True)
    assert unbatched.shape == (3, 1, 4)
    assert learned_without_position.pos_embed is None

    supplied = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
    input_token = SingleTokenLayer(
        dim=4, 
        with_pos_embed=False, 
        input_as_token=True
    )
    supplied_result = input_token((images, supplied))
    assert supplied_result.shape == (3, 1, 4)
    assert input_token.token is None and input_token.token_mlp is not None

    supplied_positioned = SingleTokenLayer(
        dim=4, 
        with_pos_embed=True, 
        input_as_token=True
    )
    assert supplied_positioned((images, supplied)).shape == (3, 1, 4)

    concatenated = SingleTokenLayer(
        dim=6, 
        with_pos_embed=True, 
        pos_merger_type="concat"
    )
    assert concatenated((images[:1], None)).shape == (1, 1, 6)
    assert concatenated.output_dim == 6
    batched_concatenated = SingleTokenLayer(
        dim=6, 
        with_pos_embed=True, 
        input_as_token=True, 
        pos_merger_type="concat"
    )
    assert batched_concatenated(
        (images, tf.ones((3, 3))),
    ).shape == (3, 1, 6)
    odd_concatenated = SingleTokenLayer(
        dim=5, 
        with_pos_embed=True, 
        pos_merger_type="concat"
    )
    assert odd_concatenated((images[:1], None)).shape == (1, 1, 4)

    projected = SingleTokenLayer(
        dim=4, 
        with_pos_embed=True, 
        embed_freq_dim=2, 
        mlp_ratio=None, 
        mlp_output_dim=None
    )
    assert projected.token_mlp is not None and projected.pos_embed_mlp is not None
    assert projected((images, None), training=True).shape == (3, 1, 4)
    projected_input = SingleTokenLayer(
        dim=6,
        with_pos_embed=True,
        input_as_token=True,
        pos_merger_type="concat",
        embed_freq_dim=2,
    )
    assert projected_input((images, tf.ones((3, 4)))).shape == (3, 1, 6)
    assert projected_input.output_dim == 6

    with tf.GradientTape() as tape:
        token_output = learned_with_position(
            (images, None), 
            training=True
        )
        loss = tf.reduce_sum(token_output)
    gradients = tape.gradient(loss, learned_with_position.trainable_variables)
    assert gradients and all(gradient is not None for gradient in gradients)

    try:
        SingleTokenLayer(dim=4, pos_merger_type="invalid")
    except ValueError:
        pass
    # This invalid case should already have raised: Invalid positional merge modes must
    # fail.
    else:
        raise AssertionError("Invalid positional merge modes must fail.")

    config = learned_with_position.get_config()
    restored = SingleTokenLayer.from_config(config)
    assert restored.dim == 4 and restored.with_pos_embed

    dtype_layer = SingleTokenLayer(
        dim=4, 
        with_pos_embed=False, 
        dtype="float64"
    )
    dtype_output = dtype_layer((tf.ones((2, 1), dtype=tf.float64), None))
    assert dtype_layer.compute_dtype == "float64"
    assert dtype_output.dtype == tf.float64

    # Without positional broadcasting, this mode legitimately carries the
    # supplied token batch independently of the image batch.
    assert input_token((images, tf.ones((2, 4)))).shape == (2, 1, 4)

    return {"SingleTokenLayer": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
