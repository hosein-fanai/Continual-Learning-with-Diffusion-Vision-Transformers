"""Base utilities for learned and sinusoidal token embeddings."""

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

from typing import Any, get_args

from . import PosEmbedType

from diffusion.layers.embedding import MergeType
from diffusion.layers.base_layer import BaseLayer


class BaseEmbedding(BaseLayer):
    """Provide positional-table construction and merge utilities.

    This class supplies common machinery rather than a public ``call`` method.
    Subclasses create ``self.pos_embed`` with :meth:`_create_embeddings` and
    merge it into token features with :meth:`_pos_merger`.

    Positional modes have the following meanings:

    * ``"new_weight"`` creates a zero-initialized learned table directly at
      the requested output size.
    * ``"1d_sincos"`` encodes flattened positions; with ``embed_steps`` it
      instead creates a rank-two lookup table for discrete conditions.
    * ``"2d_sincos"`` encodes row and column coordinates separately.
    * ``"1d_interpolate"`` and ``"2d_interpolate"`` build fixed source-grid
      tables and resize them at call time.
    * ``"1d_learned_interpolate"`` and ``"2d_learned_interpolate"`` resize
      learned source-grid tables.
    * ``None`` disables positional embeddings.

    Args:
        dim: Intended output feature width and default normalization width.
        grid_size: Positive source grid side length. It is required by spatial
            interpolation modes and normally describes the construction-time
            patch grid.
        pos_embed_type: One of :data:`PosEmbedType`, or ``None``. See the mode
            descriptions above.
        pos_interpolation_method: Method accepted by ``tf.image.resize``, such
            as ``"nearest"``, ``"bilinear"``, ``"bicubic"``, or ``"area"``.
        pos_merger_type: ``"add"`` requires equal content/position widths;
            ``"concat"`` appends position channels on the last axis.
        embed_freq_dim: Optional raw embedding width. ``None`` uses ``dim``;
            subclasses commonly project this frequency width to ``dim``.
        embed_steps: Number of discrete embedding rows, for example diffusion
            timesteps or label categories. It is required by lookup embeddings.
        embed_temperature: Positive sinusoidal wavelength temperature.
        embed_trainable: Whether a non-``"new_weight"`` lookup layer may update
            its initialized table. ``"new_weight"`` is always learned.
        **kwargs: :class:`BaseLayer` arguments such as ``use_layer_norm``,
            ``ln_mlp_ratio``, ``ln_no_adaptation``, ``mlp_ratio``,
            ``mlp_activation_func``, and ``mlp_output_dim``, plus Keras options
            including ``name``, ``dtype``, and ``trainable``. ``ln_dim`` is
            supplied internally from ``dim``; serialized configs may repeat
            only that same value.

    Inputs:
        No direct tensor input because this base class does not implement
        ``call``. Subclasses pass token tensors to :meth:`_pos_merger`.

    Outputs:
        Subclass-dependent floating tensors whose final width is controlled by
        ``dim``, the optional projection MLP, and ``pos_merger_type``.

    Serialization:
        ``from_config(get_config())`` is supported. The constructor discards
        the inherited serialized ``ln_dim`` value and derives it from ``dim``
        so those widths cannot diverge during reconstruction.
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
        **kwargs: Any
    ) -> None:
        """Store embedding configuration and validate enumerated modes.

        Args:
            dim (int): Positive target feature width.
            grid_size (int | None): Optional positive source-grid side length.
            pos_embed_type (PosEmbedType | None): Positional representation mode.
            pos_interpolation_method (str): ``tf.image.resize`` method name.
            pos_merger_type (MergeType): ``"add"`` or ``"concat"``.
            embed_freq_dim (int | None): Optional positive raw embedding width.
            embed_steps (int | None): Optional positive lookup-table row count.
            embed_temperature (float): Positive sinusoidal temperature.
            embed_trainable (bool): Whether initialized lookup tables train.
            **kwargs (Any): Typed :class:`BaseLayer` and Keras options.

        Returns:
            ``None``.
        """

        temp_val = kwargs.pop("ln_dim", dim)
        # Reject a serialized normalization width that conflicts with ``dim``.
        if temp_val is not None and temp_val != dim:
            raise ValueError("ln_dim must equal dim for embedding layers.")
        super().__init__(
            ln_dim=dim, 
            **kwargs
        )
        self._save_init_args(locals())

        # Require a positive integer target embedding width.
        if not isinstance(self.dim, int) or isinstance(self.dim, bool) or self.dim < 1:
            raise ValueError("dim must be a positive integer.")
        # Validate an optional source-grid size.
        if self.grid_size is not None and (
            not isinstance(self.grid_size, int)
            or isinstance(self.grid_size, bool)
            or self.grid_size < 1
        ):
            raise ValueError("grid_size must be None or a positive integer.")
        # Validate an optional raw frequency-embedding width.
        if self.embed_freq_dim is not None and (
            not isinstance(self.embed_freq_dim, int)
            or isinstance(self.embed_freq_dim, bool)
            or self.embed_freq_dim < 1
        ):
            raise ValueError("embed_freq_dim must be None or a positive integer.")
        # Validate an optional discrete lookup-table size.
        if self.embed_steps is not None and (
            not isinstance(self.embed_steps, int)
            or isinstance(self.embed_steps, bool)
            or self.embed_steps < 1
        ):
            raise ValueError("embed_steps must be None or a positive integer.")
        # Require a positive sinusoidal wavelength temperature.
        if self.embed_temperature <= 0:
            raise ValueError("embed_temperature must be positive.")
        # Restrict positional construction to the documented modes.
        if self.pos_embed_type not in (None,) + get_args(PosEmbedType):
            raise ValueError(
                f"pos_embed_type must be None or one of {get_args(PosEmbedType)}."
            )
        # Restrict positional merging to addition or concatenation.
        if self.pos_merger_type not in get_args(MergeType):
            raise ValueError(
                f"pos_merger_type must be one of {get_args(MergeType)}."
            )


        self.embed_dim = self.dim if self.embed_freq_dim is None else self.embed_freq_dim
        self.pos_embed_mlp = None

    def _get_1d_sincos_embedding(
        self, 
        dim: int, 
        positions: np.ndarray, 
        temperature: float = 10_000.
    ) -> np.ndarray:
        """Return a one-dimensional sinusoidal table.

        Args:
            dim (int): Positive output width. Odd values are supported by truncating
                the final cosine channel.
            positions (np.ndarray): Rank-one NumPy array or eagerly convertible tensor of
                numeric positions, shaped ``[count]``.
            temperature (float): Positive wavelength temperature.

        Returns:
            ``np.ndarray`` shaped ``[count, dim]``. Values are sine/cosine
            features and are normally ``float32``.
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

    def _get_2d_sincos_embedding(
        self, 
        dim: int, 
        grid_size: int, 
        temperature: float = 10_000., 
        name: str | None = None
    ) -> tf.Tensor:
        """Build a fixed row-major 2D sine/cosine positional table.

        Unlike the common implementation that requires ``dim % 4 == 0``, this
        function supports every positive channel dimension by splitting the
        channels between the horizontal and vertical coordinates and truncating
        the final sine/cosine pair when necessary.

        Args:
            dim (int): Positive output channel count.
            grid_size (int): Positive square-grid side length.
            temperature (float): Positive wavelength temperature shared by both axes.
            name (str | None): Optional TensorFlow operation name.

        Returns:
            ``tf.Tensor`` of dtype ``tf.float32`` and shape
            ``[1, grid_size * grid_size, dim]`` in row-major order.
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

    def _get_t_embedding(
        self, 
        t: tf.Tensor | np.ndarray, 
        dim: int
    ) -> tf.Tensor: # old
        """Build the legacy one-dimensional timestep encoding.

        Args:
            t (tf.Tensor | np.ndarray): Rank-one numeric values shaped ``[batch]``.
            dim (int): Requested integer width. The legacy algorithm returns
                ``2 * (dim // 2)`` channels, so odd values lose one channel.

        Returns:
            ``tf.Tensor`` shaped ``[batch, 2 * (dim // 2)]``. This helper is
            retained for compatibility; new code uses
            :meth:`_get_1d_sincos_embedding`.
        """

        half = dim // 2
        freqs = np.exp(-np.log(10_000) * np.arange(half) / half)
        args = t[:, None] * freqs[None]

        emb = tf.concat([
            tf.sin(args), 
            tf.cos(args)
        ], axis=-1)

        return emb

    def _get_2d_pos_embed(self, h: int, w: int, dim: int) -> tf.Tensor: # old
        """Build the legacy flattened two-dimensional sinusoidal table.

        Args:
            h (int): Positive integer grid height.
            w (int): Positive integer grid width.
            dim (int): Requested channel width. For exact width it must be a positive
                multiple of four; otherwise the result is truncated to
                ``4 * (dim // 4)`` channels.

        Returns:
            ``tf.Tensor`` of dtype ``tf.float32`` shaped
            ``[1, h * w, 4 * (dim // 4)]``. This compatibility helper is not
            used by the current positional modes.
        """

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
    ) -> tf.Variable | tf.Tensor | np.ndarray | None:
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

        Args:
            embed_dim (int | None): Table width; ``None`` uses ``self.embed_dim``.
            grid_size (int | None): Source square-grid size; ``None`` uses
                ``self.grid_size``. Required by interpolation modes.
            output_grid_size (int | None): Target square-grid size. Required by spatial
                ``new_weight`` and sine/cosine modes.
            pos_embed_type (PosEmbedType | None): Positional mode override. ``None`` inherits the
                instance mode; when the instance mode is also ``None``, this
                method returns ``None``.
            embed_steps (int | None): Number of rows for a non-spatial ``"1d_sincos"``
                condition table. ``None`` makes that mode spatial with
                ``output_grid_size ** 2`` rows.
            temperature (float | None): Positive sinusoidal temperature; ``None`` uses the
                configured value.
            name (str | None): Optional weight or tensor name.

        Returns:
            ``tf.Variable | tf.Tensor | np.ndarray | None``. Spatial results
            have shape ``[1, positions, embed_dim]`` except the internal 2-D
            learned source table ``[1, grid, grid, embed_dim]``. A condition
            ``1d_sincos`` table has shape ``[embed_steps, embed_dim]``.
        """

        embed_dim = self.embed_dim if embed_dim is None else embed_dim
        grid_size = self.grid_size if grid_size is None else grid_size
        pos_embed_type = None if pos_embed_type is None and self.pos_embed_type is None \
                        else pos_embed_type or self.pos_embed_type
        embed_steps = self.embed_steps if embed_steps is None else embed_steps
        temperature = self.embed_temperature if temperature is None else temperature
        name = f"{self.name}/positional_embeddings" if name is None else name

        self.pos_embed_type = pos_embed_type

        # Disable positional embeddings when no construction mode is selected.
        if pos_embed_type is None:
            return None

        # Create a trainable table directly at the target resolution.
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

        # Build a fixed one-dimensional sinusoidal lookup or spatial table.
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

        # Build the fixed one-dimensional source table for later resizing.
        if pos_embed_type == "1d_interpolate":
            return self._get_1d_sincos_embedding(
                dim=embed_dim,
                positions=tf.range(
                    0, grid_size * grid_size, 1, dtype=tf.float32
                ),
                temperature=temperature
            )[None, ...]

        # Build a trainable one-dimensional source table for later resizing.
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

        # Build a fixed two-dimensional table at the target resolution.
        if pos_embed_type == "2d_sincos":
            return self._get_2d_sincos_embedding(
                dim=embed_dim, grid_size=output_grid_size, 
                temperature=temperature, name=name
            )

        # Build the fixed two-dimensional source table for later resizing.
        if pos_embed_type == "2d_interpolate":
            return self._get_2d_sincos_embedding(
                dim=embed_dim, grid_size=grid_size, 
                temperature=temperature, name=name
            )

        # Build a trainable two-dimensional source grid for later resizing.
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

    def _create_embedding_layer(self, **kwargs: Any) -> layers.Embedding:
        """Create a discrete Keras embedding initialized by this base class.

        This factory is intended for :class:`ConditionEmbedding`. Effective
        keyword keys are ``pos_embed_type``, ``embed_steps``, ``embed_dim``,
        ``grid_size``, ``output_grid_size``, ``temperature``, and ``name``;
        omitted values inherit instance attributes. For a valid rank-two lookup
        table, use ``pos_embed_type="new_weight"`` or
        ``pos_embed_type="1d_sincos"`` with ``embed_steps`` set. Configure
        ``embed_trainable`` on the constructor rather than passing it here:
        non-new modes forward keyword arguments to
        :meth:`_create_embeddings`, which does not accept that key.

        Example::

            layer._create_embedding_layer(
                pos_embed_type="1d_sincos",
                embed_steps=1000,
                embed_dim=64,
                temperature=10_000.0,
            )

        Args:
            **kwargs (Any): Overrides listed above. Unknown keys are invalid for
                initialized non-new tables and may raise ``TypeError``.

        Returns:
            ``tf.keras.layers.Embedding`` mapping integer tensors of arbitrary
            shape to that shape plus a final ``embed_dim`` axis. A
            ``"new_weight"`` layer is always trainable; other modes honor the
            instance's ``embed_trainable`` flag.
        """

        kwargs["pos_embed_type"] = None if kwargs.get("pos_embed_type", None) is None \
                                and self.pos_embed_type is None \
                                else kwargs.get("pos_embed_type", None) or self.pos_embed_type
        embedding_layer = layers.Embedding(
            input_dim=kwargs.get("embed_steps", self.embed_steps), 
            output_dim=kwargs.get("embed_dim", self.embed_dim), 
            trainable=kwargs.get("embed_trainable", self.embed_trainable), 
            name="embeddings",
            dtype=self.dtype_policy,
        )

        # Install a deterministic initializer for non-random embedding modes.
        if kwargs["pos_embed_type"] != "new_weight":
            embedding_layer.build(())
            embedding_layer.set_weights([self._create_embeddings(**kwargs)])
        # Keep directly initialized new-weight embeddings trainable.
        else:
            embedding_layer.trainable = True

        return embedding_layer

    def _pos_merger(
        self, 
        x: tf.Tensor, 
        batch_size: int | tf.Tensor | None = None,
        output_grid_size: int | None = None, 
        training: bool | tf.Tensor | None = None
    ) -> tf.Tensor:
        """Resolve, batch-broadcast, and merge a spatial positional embedding.

        Args:
            x (tf.Tensor): Floating content tensor shaped ``[batch, tokens, channels]``.
            batch_size (int | tf.Tensor | None): Scalar integer used to repeat
                the positional table. ``None`` reads it from ``x``.
            output_grid_size (int | None): Optional target grid side. Supplying it resizes
                even a non-interpolation table; interpolation modes use it or
                the subclass's ``self.output_grid_size``.
            training (bool | tf.Tensor | None): Optional Keras training flag forwarded to the positional
                projection MLP.

        Returns:
            ``tf.Tensor``. With no positional mode, this is ``x`` unchanged.
            ``"add"`` preserves shape and requires equal last dimensions;
            ``"concat"`` returns ``[batch, tokens, channels + pos_channels]``.
            Spatial token count must equal ``output_grid_size ** 2``.
        """

        # Preserve content unchanged when positional embeddings are disabled.
        if self.pos_embed_type is None:
            return x

        batch_size = tf.shape(x)[0] if batch_size is None else batch_size
        pos_embed = self.pos_embed

        interpolated_pos_embed = "interpolate" in self.pos_embed_type
        # Resize whenever explicitly requested or required by an interpolation mode.
        if output_grid_size is not None or interpolated_pos_embed:
            output_grid_size = self.output_grid_size if output_grid_size is None \
                            else output_grid_size
            source_grid_size = self.grid_size if interpolated_pos_embed \
                             else self.output_grid_size

            pos_embed_dim = pos_embed.shape[-1]
            # Treat flattened one-dimensional tables as tall single-column images.
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
            # Resize two-dimensional tables in their native square layout.
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

        # Append positional channels for concatenation mode.
        if self.pos_merger_type == "concat":
            return tf.concat([x, pos_embed], axis=-1)

        # Add position features elementwise for additive mode.
        if self.pos_merger_type == "add":
            return x + pos_embed


def run_self_tests() -> dict[str, str]:
    """Exercise every embedding-table and positional-merge mode.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"BaseEmbedding": "passed"}`` after sinusoidal, learned,
        interpolated, lookup, merge, dtype, validation, and config checks.
    """

    import numpy as np


    for invalid_type in ("rotary", "", 1):
        try:
            BaseEmbedding(dim=4, pos_embed_type=invalid_type)
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown positional embedding modes must fail.")
    for invalid_merge in ("multiply", "", None):
        try:
            BaseEmbedding(dim=4, pos_merger_type=invalid_merge)
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown positional merge modes must fail.")

    helper = BaseEmbedding(dim=5, pos_embed_type=None)
    positions = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    for width in (1, 2, 5):
        embedding_1d = helper._get_1d_sincos_embedding(width, positions, 100.0)
        assert embedding_1d.shape == (3, width)
        assert np.isfinite(embedding_1d).all()
    embedding_2d = helper._get_2d_sincos_embedding(5, 2, temperature=100.0)
    assert embedding_2d.shape == (1, 4, 5)
    assert embedding_2d.dtype == tf.float32
    assert helper._get_t_embedding(tf.constant([0.0, 1.0]), 5).shape == (2, 4)
    assert helper._get_2d_pos_embed(2, 3, 6).shape == (1, 6, 4)

    specifications = {
        "new_weight": (1, 9, 4), 
        "1d_sincos": (1, 9, 4), 
        "1d_interpolate": (1, 4, 4), 
        "1d_learned_interpolate": (1, 4, 4), 
        "2d_sincos": (1, 9, 4), 
        "2d_interpolate": (1, 4, 4), 
        "2d_learned_interpolate": (1, 2, 2, 4), 
    }
    for mode, expected_shape in specifications.items():
        table_owner = BaseEmbedding(dim=4, grid_size=2, pos_embed_type=mode)
        table = table_owner._create_embeddings(output_grid_size=3)
        assert tuple(table.shape) == expected_shape, (mode, table.shape)
        # Learned modes must expose their table as a trainable variable.
        if "learned" in mode or mode == "new_weight":
            assert any(variable is table for variable in table_owner.trainable_variables)
        # Fixed sinusoidal modes must not introduce trainable variables.
        else:
            assert not table_owner.trainable_variables

    disabled = BaseEmbedding(dim=4, pos_embed_type=None)
    assert disabled._create_embeddings(output_grid_size=2) is None
    identity_input = tf.ones((2, 4, 4))
    assert disabled._pos_merger(identity_input) is identity_input

    condition_owner = BaseEmbedding(
        dim=4, 
        pos_embed_type="1d_sincos", 
        embed_steps=5
    )
    condition_table = condition_owner._create_embeddings(output_grid_size=None)
    assert condition_table.shape == (5, 4)

    learned_lookup_owner = BaseEmbedding(
        dim=3, 
        pos_embed_type="new_weight", 
        embed_steps=4,
        embed_trainable=False
    )
    learned_lookup = learned_lookup_owner._create_embedding_layer()
    assert learned_lookup.trainable
    assert learned_lookup(tf.constant([[0, 3]], tf.int32)).shape == (1, 2, 3)

    for trainable in (False, True):
        fixed_lookup_owner = BaseEmbedding(
            dim=4, pos_embed_type="1d_sincos", 
            embed_steps=4,
            embed_trainable=trainable,
        )
        fixed_lookup = fixed_lookup_owner._create_embedding_layer()
        assert fixed_lookup.trainable is trainable
        assert fixed_lookup(tf.constant([0, 1, 3], tf.int64)).shape == (3, 4)
    try:
        out_of_range = fixed_lookup(tf.constant([4], tf.int32))
    except tf.errors.InvalidArgumentError:
        pass
    else:
        # TensorFlow documents device-dependent behavior for out-of-range
        # Embedding gathers; some GPU kernels return a finite row.
        assert out_of_range.shape == (1, 4)
        assert np.isfinite(out_of_range.numpy()).all()

    for mode in ("1d_interpolate", "2d_interpolate",
                 "1d_learned_interpolate", "2d_learned_interpolate"):
        for interpolation in (
            "nearest", "bilinear", "bicubic", "area", "lanczos3",
            "lanczos5", "gaussian", "mitchellcubic",
        ):
            merger = BaseEmbedding(
                dim=4, 
                grid_size=2, 
                pos_embed_type=mode, 
                pos_interpolation_method=interpolation, 
                pos_merger_type="add"
            )
            merger.output_grid_size = 3
            merger.pos_embed = merger._create_embeddings(output_grid_size=3)
            merged = merger._pos_merger(tf.ones((2, 9, 4)), training=False)
            assert merged.shape == (2, 9, 4)

    invalid_interpolation = BaseEmbedding(
        dim=4, 
        grid_size=2, 
        pos_embed_type="2d_interpolate", 
        pos_interpolation_method="not-a-resizer"
    )
    invalid_interpolation.output_grid_size = 3
    invalid_interpolation.pos_embed = invalid_interpolation._create_embeddings(
        output_grid_size=3,
    )
    try:
        invalid_interpolation._pos_merger(tf.ones((1, 9, 4)))
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown TensorFlow resize methods must fail.")

    additive = BaseEmbedding(
        dim=4, grid_size=2, 
        pos_embed_type="new_weight", 
        pos_merger_type="add"
    )
    additive.output_grid_size = 2
    additive.pos_embed = additive._create_embeddings(output_grid_size=2)
    assert additive._pos_merger(tf.ones((2, 4, 4))).shape == (2, 4, 4)
    assert additive._pos_merger(
        tf.ones((2, 9, 4)), 
        output_grid_size=3, 
    ).shape == (2, 9, 4)

    concatenating = BaseEmbedding(
        dim=2, grid_size=2, 
        pos_embed_type="2d_sincos",
        pos_merger_type="concat"
    )
    concatenating.output_grid_size = 2
    concatenating.pos_embed = concatenating._create_embeddings(output_grid_size=2)
    assert concatenating._pos_merger(tf.ones((3, 4, 2))).shape == (3, 4, 4)

    try:
        additive._pos_merger(tf.ones((1, 3, 4)))
    except (tf.errors.InvalidArgumentError, ValueError):
        pass
    else:
        raise AssertionError("Mismatched content and position counts must fail.")

    config = BaseEmbedding(dim=4, grid_size=2).get_config()
    restored = BaseEmbedding.from_config(config)
    assert restored.dim == 4 and restored.grid_size == 2

    dtype_fixed = BaseEmbedding(
        dim=4, pos_embed_type="2d_sincos", dtype="float64",
    )
    fixed_table = dtype_fixed._create_embeddings(output_grid_size=2)
    assert dtype_fixed.compute_dtype == "float64"
    assert fixed_table.dtype == tf.float32
    dtype_learned = BaseEmbedding(
        dim=4, pos_embed_type="new_weight", dtype="float64",
    )
    learned_table = dtype_learned._create_embeddings(output_grid_size=2)
    assert learned_table.dtype == tf.float64

    return {"BaseEmbedding": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
