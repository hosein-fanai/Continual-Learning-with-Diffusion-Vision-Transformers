"""Conditional convolutional U-Net compatible with diffusion wrappers."""

import tensorflow as tf
from tensorflow.keras import layers

from collections.abc import Sequence
from typing import NoReturn

from . import UNetInputs, DTypeLike, UNetFullOutput

from common.argument_saver import ArgumentSaverModel

from diffusion.layers.embedding.condition_embedding import ConditionEmbedding


class _ResidualBlock(layers.Layer):
    """Apply two convolutions and add a shape-compatible residual path.

    The residual path is left unchanged when the input already has ``width``
    channels. Otherwise, a learned 1x1 convolution projects it to the required
    width. This retains the behavior of the original U-Net while making the
    block a reusable Keras layer whose variables are created once.

    Args:
        width: Number of output channels produced by both spatial
            convolutions.
        activation_func: Keras activation used by the first convolution.
        use_batch_norm: Whether to normalize the input before the
            convolutions.
        name: Optional Keras layer name.
        dtype: Optional Keras computation and variable dtype policy.

    Inputs:
        Floating channels-last feature tensor
        ``[batch, height, width, channels]``.

    Outputs:
        Floating tensor ``[batch, height, width, width]``.
    """

    def __init__(
        self,
        width: int,
        activation_func: str = "swish",
        use_batch_norm: bool = True,
        name: str | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        """Initialize residual projections and two spatial convolutions.

        Args and accepted data types are defined in the class documentation.

        Returns:
            ``None``. Variables whose shape depends on input channels are
            completed by :meth:`build`.
        """

        super().__init__(name=name, dtype=dtype)
        self.width = width
        self.activation_func = activation_func
        self.use_batch_norm = use_batch_norm

        self.normalization = layers.BatchNormalization(
            center=False,
            scale=False,
            dtype=self.dtype_policy,
            name=f"{self.name}/normalization",
        ) if self.use_batch_norm else None
        self.first_convolution = layers.Conv2D(
            filters=self.width,
            kernel_size=3,
            padding="same",
            activation=self.activation_func,
            dtype=self.dtype_policy,
            name=f"{self.name}/first_convolution",
        )
        self.second_convolution = layers.Conv2D(
            filters=self.width,
            kernel_size=3,
            padding="same",
            dtype=self.dtype_policy,
            name=f"{self.name}/second_convolution",
        )
        self.residual_projector: layers.Conv2D | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create a residual projection only when channel widths differ.

        Args:
            input_shape: Shape of the feature map that will enter the block.

        Returns:
            ``None``. The method creates the optional projection layer and
            marks this residual block as built.
        """

        input_width = tf.TensorShape(input_shape)[-1]
        if input_width is None:
            raise ValueError("A residual block requires a known channel size.")

        if int(input_width) != self.width:
            self.residual_projector = layers.Conv2D(
                filters=self.width,
                kernel_size=1,
                dtype=self.dtype_policy,
                name=f"{self.name}/residual_projector",
            )

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
    ) -> tf.Tensor:
        """Transform a feature map and add its residual representation.

        Args:
            inputs: Image-like tensor shaped ``[batch, height, width,
                channels]``.
            training: Keras training flag forwarded to batch normalization.

        Returns:
            A tensor shaped ``[batch, height, width, width]``, where the last
            ``width`` refers to this block's configured channel count.
        """

        residual = self.residual_projector(
            inputs,
            training=training,
        ) if self.residual_projector is not None else inputs
        x = self.normalization(
            inputs,
            training=training,
        ) if self.normalization is not None else inputs
        x = self.first_convolution(x, training=training)
        x = self.second_convolution(x, training=training)

        return x + residual


class _DownBlock(layers.Layer):
    """Run residual blocks, retain their outputs, and reduce spatial size.

    Args:
        width: Channel width used by every residual block at this level.
        block_depth: Number of residual blocks and therefore skip tensors at
            this level.
        activation_func: Keras activation used inside residual blocks.
        use_batch_norm: Whether residual blocks use batch normalization.
        name: Optional Keras layer name.
        dtype: Optional Keras computation and variable dtype policy.

    Inputs:
        Floating channels-last feature tensor
        ``[batch, height, width, channels]``.

    Outputs:
        Pair of a downsampled floating feature tensor and an ordered
        ``list[tf.Tensor]`` of full-resolution skips.
    """

    def __init__(
        self,
        width: int,
        block_depth: int,
        activation_func: str = "swish",
        use_batch_norm: bool = True,
        name: str | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        """Create one encoder level and its fixed 2x average pool.

        Args and accepted data types are defined in the class documentation.

        Returns:
            ``None``.
        """

        super().__init__(name=name, dtype=dtype)
        self.residual_blocks = [
            _ResidualBlock(
                width=width,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                dtype=self.dtype_policy,
                name=f"{self.name}/residual_{block_id + 1}",
            )
            for block_id in range(block_depth)
        ]
        self.downsampler = layers.AveragePooling2D(
            pool_size=2,
            padding="same",
            dtype=self.dtype_policy,
            name=f"{self.name}/downsampler",
        )

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        """Encode one level and return its downsampled output and skips.

        Args:
            inputs: Feature tensor shaped ``[batch, height, width, channels]``.
            training: Keras training flag forwarded to residual blocks.

        Returns:
            A pair containing the spatially downsampled feature tensor and the
            ordered residual outputs that the decoder will consume as skips.
        """

        x = inputs
        skips = []
        for residual_block in self.residual_blocks:
            x = residual_block(x, training=training)
            skips.append(x)

        return self.downsampler(x), skips


class _UpBlock(layers.Layer):
    """Upsample one decoder level and consume encoder skips in LIFO order.

    Args:
        width: Channel width produced by every residual block at this level.
        block_depth: Number of skip tensors consumed at this level.
        activation_func: Keras activation used inside residual blocks.
        use_batch_norm: Whether residual blocks use batch normalization.
        interpolation: Image-resize interpolation used by the decoder.
        name: Optional Keras layer name.
        dtype: Optional Keras computation and variable dtype policy.

    Inputs:
        Pair ``(decoder, skips)`` where ``decoder`` is a floating channels-last
        tensor and ``skips`` is ``list[tf.Tensor]`` at the target spatial size.

    Outputs:
        Floating decoded tensor at the skip height/width and configured width.
    """

    def __init__(
        self,
        width: int,
        block_depth: int,
        activation_func: str = "swish",
        use_batch_norm: bool = True,
        interpolation: str = "bilinear",
        name: str | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        """Create one decoder level and its residual blocks.

        Args and accepted data types are defined in the class documentation.

        Returns:
            ``None``.
        """

        super().__init__(name=name, dtype=dtype)
        self.block_depth = block_depth
        self.interpolation = interpolation
        self.residual_blocks = [
            _ResidualBlock(
                width=width,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                dtype=self.dtype_policy,
                name=f"{self.name}/residual_{block_id + 1}",
            )
            for block_id in range(block_depth)
        ]

    def call(
        self,
        inputs: tuple[tf.Tensor, list[tf.Tensor]],
        training: bool | None = None,
    ) -> tf.Tensor:
        """Decode one level using the most recently stored skips first.

        The decoder tensor is resized once to this level's exact skip shape.
        This makes the U-Net work with odd and progressively changed image
        resolutions without applying an extra alignment interpolation.

        Args:
            inputs: Pair of the current decoder tensor and this level's encoder
                skip tensors. The skips are read in reverse order.
            training: Keras training flag forwarded to residual blocks.

        Returns:
            The decoded feature tensor at the spatial size of this level's
            encoder features.
        """

        x, skips = inputs
        x = tf.image.resize(
            x,
            size=tf.shape(skips[-1])[1:3],
            method=self.interpolation,
        )
        x = tf.cast(x, skips[-1].dtype)

        for residual_block, skip in zip(
            self.residual_blocks,
            reversed(skips),
        ):
            x = tf.concat([x, skip], axis=-1)
            x = residual_block(x, training=training)

        return x


class UNet(ArgumentSaverModel):
    """A conditional convolutional U-Net for diffusion noise prediction.

    ``UNet`` modernizes the architecture previously stored in this module and
    implements the ``DiffusionModel`` call and serialization contract used by
    ``DiffusionTransformer``. It takes a noisy image, an integer diffusion
    timestep, and an integer class label; embeds both conditions; predicts the
    noise component of the image; and can be passed directly to
    ``diffusion.models.wrapper.diffusion_model``'s ``DiffusionModel``.

    The encoder applies residual blocks at each configured channel width and
    stores every residual output as a skip. The bottleneck processes the most
    compressed representation. The decoder then consumes the skips in reverse
    order, resizing with the configured interpolation method to each skip's
    exact size. Consequently the network accepts progressive resolutions below,
    above, or unrelated to its construction-time ``image_size`` and always
    returns the input spatial size. The final 1x1 convolution is
    zero-initialized, as is common for diffusion noise heads.

    Classifier-free guidance is coordinated by ``DiffusionModel``. When
    ``use_cfg=True``, embedding index 0 represents the unconditional class and
    real classes use indices 1 through ``num_classes``. During training and
    evaluation, callers provide ordinary zero-based dataset labels and the
    wrapper performs this shift. Direct network calls and ``DiffusionModel``
    sampling use embedding indices, so use 0 for unconditional generation and
    1 through ``num_classes`` for real classes. The wrapper combines conditional
    and unconditional predictions when guidance is requested.

    Args:
        num_classes: Number of real dataset classes. One additional embedding
            is created for the unconditional class when ``use_cfg`` is true.
        use_cfg: Whether the network reserves label 0 for classifier-free
            guidance. Guidance evaluation itself remains in ``DiffusionModel``.
        timesteps: Number of integer diffusion steps and entries in the fixed
            sinusoidal timestep embedding table.
        image_size: Native square image size. This initializes
            ``current_resolution`` but does not constrain later progressive
            resolutions.
        channels: Number of channels in both noisy-image inputs and predicted
            noise outputs.
        widths: Encoder channel widths, ordered from highest to lowest spatial
            resolution. The decoder uses the reverse order.
        block_depth: Number of residual blocks created per encoder and decoder
            level. Every encoder block contributes one skip tensor.
        bottleneck_width: Channel width used between the encoder and decoder.
        bottleneck_depth: Number of residual blocks in the bottleneck.
        image_embedding_dim: Channels used to project the noisy image before
            concatenating its conditions. The original implementation used 21.
        time_embedding_dim: Width of the fixed sinusoidal timestep embedding.
            The original implementation used 22.
        label_embedding_dim: Width of the learned class embedding. The original
            implementation used 21.
        activation_func: Keras activation used by residual-block convolutions.
        final_activation_func: Keras activation applied to predicted noise.
            ``"linear"`` leaves the prediction unconstrained.
        use_batch_norm: Whether residual blocks apply batch normalization.
        upsampling_interpolation: Resize method used in decoder levels.
        name_prefix: Prefix added to internally created layer names.
        build: Whether to build all variables immediately. Keep this enabled
            when wrapping the model with ``DiffusionModel`` so its EMA clone can
            copy weights immediately.
        **kwargs: Standard Keras model arguments such as ``name``, ``dtype``,
            and ``trainable``.

    Inputs:
        A tuple ``(noisy_images, timesteps, labels)``. Images have shape
        ``[batch, height, width, channels]``; timesteps and labels have shape
        ``[batch]``.

    Outputs:
        By default, predicted noise with the same shape as ``noisy_images``.
        With ``full_return=True``, returns the five-item auxiliary structure
        expected by ``DiffusionModel``. This plain U-Net has no KL or token
        regularizers, so their entries are compatibility placeholders.
    """

    def __init__(
        self,
        num_classes: int = 10,
        use_cfg: bool = True,
        timesteps: int = 1_000,
        image_size: int = 32,
        channels: int = 1,
        widths: Sequence[int] = (32, 64, 96),
        block_depth: int = 2,
        bottleneck_width: int = 128,
        bottleneck_depth: int = 2,
        image_embedding_dim: int = 21,
        time_embedding_dim: int = 22,
        label_embedding_dim: int = 21,
        activation_func: str = "swish",
        final_activation_func: str = "linear",
        use_batch_norm: bool = True,
        upsampling_interpolation: str = "bilinear",
        name_prefix: str = "",
        build: bool = True,
        **kwargs,
    ) -> None:
        """Construct, validate, and optionally build the conditional U-Net.

        The full argument contract—including allowed values, Keras ``**kwargs``,
        input tensor dtypes/shapes, and output structure—is documented on
        :class:`UNet`.

        Returns:
            ``None``. When ``build=True``, all nested variables and symbolic
            input/output tensors are created before construction returns.
        """

        widths = tuple(widths)
        super().__init__(**kwargs)
        self._check_arguments(
            num_classes=num_classes,
            timesteps=timesteps,
            image_size=image_size,
            channels=channels,
            widths=widths,
            block_depth=block_depth,
            bottleneck_width=bottleneck_width,
            bottleneck_depth=bottleneck_depth,
            image_embedding_dim=image_embedding_dim,
            time_embedding_dim=time_embedding_dim,
            label_embedding_dim=label_embedding_dim,
        )
        self._save_init_args(locals())
        self._init_config.update({
            "name": self.name,
            "trainable": self.trainable,
            "dtype": self.dtype_policy.name,
        })

        self.num_labels = self.num_classes + int(self.use_cfg)
        self.depth = len(self.widths)
        self.reshaper_kwargs = {}
        self.cls_token_regularizer_ids = []
        self.set_current_resolution()

        self.image_embedder = layers.Conv2D(
            filters=self.image_embedding_dim,
            kernel_size=1,
            dtype=self.dtype_policy,
            name=f"{self.name_prefix}image_embedder",
        )
        self.time_embedder = ConditionEmbedding(
            dim=self.time_embedding_dim,
            pos_embed_type="1d_sincos",
            embed_steps=self.timesteps,
            embed_trainable=False,
            dtype=self.dtype_policy,
            name=f"{self.name_prefix}time_embedder",
        )
        self.label_embedder = ConditionEmbedding(
            dim=self.label_embedding_dim,
            pos_embed_type="new_weight",
            embed_steps=self.num_labels,
            embed_trainable=True,
            dtype=self.dtype_policy,
            name=f"{self.name_prefix}label_embedder",
        )
        self.encoder_blocks = [
            _DownBlock(
                width=width,
                block_depth=self.block_depth,
                activation_func=self.activation_func,
                use_batch_norm=self.use_batch_norm,
                dtype=self.dtype_policy,
                name=f"{self.name_prefix}encoder_{level + 1}",
            )
            for level, width in enumerate(self.widths)
        ]
        self.bottleneck_blocks = [
            _ResidualBlock(
                width=self.bottleneck_width,
                activation_func=self.activation_func,
                use_batch_norm=self.use_batch_norm,
                dtype=self.dtype_policy,
                name=f"{self.name_prefix}bottleneck_{block_id + 1}",
            )
            for block_id in range(self.bottleneck_depth)
        ]
        self.decoder_blocks = [
            _UpBlock(
                width=width,
                block_depth=self.block_depth,
                activation_func=self.activation_func,
                use_batch_norm=self.use_batch_norm,
                interpolation=self.upsampling_interpolation,
                dtype=self.dtype_policy,
                name=f"{self.name_prefix}decoder_{level + 1}",
            )
            for level, width in enumerate(reversed(self.widths))
        ]
        self.output_projection = layers.Conv2D(
            filters=self.channels,
            kernel_size=1,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name=f"{self.name_prefix}noise_projection",
        )
        self.output_activation = layers.Activation(
            self.final_activation_func,
            dtype=self.dtype_policy,
            name=f"{self.name_prefix}predicted_noise",
        )

        if self.build_:
            self.build()

    @staticmethod
    def _check_arguments(
        num_classes: int,
        timesteps: int,
        image_size: int,
        channels: int,
        widths: tuple[int, ...],
        block_depth: int,
        bottleneck_width: int,
        bottleneck_depth: int,
        image_embedding_dim: int,
        time_embedding_dim: int,
        label_embedding_dim: int,
    ) -> None:
        """Validate dimensions that determine the U-Net's variable shapes.

        Args:
            num_classes: Number of real classes.
            timesteps: Number of diffusion timesteps.
            image_size: Native square image size.
            channels: Image and output channel count.
            widths: Encoder channel widths.
            block_depth: Residual blocks per encoder/decoder level.
            bottleneck_width: Bottleneck channel width.
            bottleneck_depth: Number of bottleneck residual blocks.
            image_embedding_dim: Noisy-image projection width.
            time_embedding_dim: Timestep embedding width.
            label_embedding_dim: Class embedding width.

        Returns:
            ``None``. Invalid non-positive dimensions raise ``ValueError``.
        """

        dimensions = {
            "num_classes": num_classes,
            "timesteps": timesteps,
            "image_size": image_size,
            "channels": channels,
            "block_depth": block_depth,
            "bottleneck_width": bottleneck_width,
            "bottleneck_depth": bottleneck_depth,
            "image_embedding_dim": image_embedding_dim,
            "time_embedding_dim": time_embedding_dim,
            "label_embedding_dim": label_embedding_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        if len(widths) == 0:
            raise ValueError("widths must contain at least one encoder width.")
        if any(
            not isinstance(width, int) or isinstance(width, bool) or width < 1
            for width in widths
        ):
            raise ValueError("Every item in widths must be a positive integer.")

    def _broadcast_condition(
        self,
        condition: tf.Tensor,
        images: tf.Tensor,
    ) -> tf.Tensor:
        """Broadcast per-example conditions across an image's spatial axes.

        Args:
            condition: Tensor shaped ``[batch, condition_channels]``.
            images: Image-like tensor whose dynamic height and width determine
                the broadcast target.

        Returns:
            Condition map shaped ``[batch, height, width,
            condition_channels]``.
        """

        condition_dim = self.time_embedding_dim + self.label_embedding_dim
        condition = tf.cast(condition, images.dtype)
        condition = tf.reshape(condition, (-1, 1, 1, condition_dim))
        condition = tf.broadcast_to(
            condition,
            shape=tf.concat([
                tf.shape(images)[:3],
                tf.constant([condition_dim], dtype=tf.int32),
            ], axis=0),
        )
        condition.set_shape((None, None, None, condition_dim))

        return condition

    @property
    def current_resolution(self) -> int:
        """Return the square image resolution currently processed.

        Returns:
            The active positive integer resolution.
        """

        return self._current_resolution

    def build(
        self,
        input_shape: tuple[tf.TensorShape, tf.TensorShape, tf.TensorShape]
        | None = None,
    ) -> None:
        """Build every U-Net variable using the active resolution.

        ``input_shape`` is accepted for normal Keras compatibility. The model's
        canonical three-input shapes are created by ``build_model`` so that an
        EMA clone has exactly the same variables and weight ordering.

        Args:
            input_shape: Optional Keras-provided input shape. It does not alter
                the configured image, timestep, or label contracts.

        Returns:
            ``None``. All nested layers are built as a side effect.
        """

        del input_shape
        model_input_shapes = self.build_model()
        super().build(model_input_shapes)

    def call(
        self,
        inputs: UNetInputs,
        full_return: bool = False,
        training: bool | None = None,
    ) -> tf.Tensor | UNetFullOutput:
        """Predict diffusion noise from an image, timestep, and class label.

        Args:
            inputs: Tuple ``(noisy_images, timesteps, labels)``. Images are a
                rank-four floating tensor; timesteps and labels are rank-one
                integer tensors. Timestep values must be in
                ``[0, timesteps)`` and label values in ``[0, num_labels)``.
            full_return: When false, return only predicted noise. When true,
                also return the condition embedding and compatibility
                placeholders consumed by ``DiffusionModel``.
            training: Keras training flag forwarded to batch normalization.

        Returns:
            Predicted noise matching ``noisy_images``. With ``full_return``,
            returns ``(noise, condition, [], [None], (None, None))``. The empty
            feature and regularization placeholders indicate that this plain
            U-Net has no transformer features, token regularizer, or KL latent
            state.
        """

        noisy_images, timesteps, labels = inputs
        time_embeddings = self.time_embedder(timesteps, training=training)
        label_embeddings = self.label_embedder(labels, training=training)
        condition = tf.concat(
            [time_embeddings, label_embeddings],
            axis=-1,
        )
        condition = tf.cast(condition, self.compute_dtype)

        x = self.image_embedder(noisy_images, training=training)
        condition_map = self._broadcast_condition(condition, x)
        x = tf.concat([x, condition_map], axis=-1)

        skips = []
        for encoder_block in self.encoder_blocks:
            x, level_skips = encoder_block(x, training=training)
            skips.extend(level_skips)

        for bottleneck_block in self.bottleneck_blocks:
            x = bottleneck_block(x, training=training)

        for decoder_block in self.decoder_blocks:
            level_skips = skips[-decoder_block.block_depth:]
            skips = skips[:-decoder_block.block_depth]
            x = decoder_block((x, level_skips), training=training)

        predicted_noise = self.output_projection(x, training=training)
        predicted_noise = self.output_activation(predicted_noise)

        if full_return:
            return (
                predicted_noise,
                condition,
                [],
                [None],
                (None, None),
            )

        return predicted_noise

    def set_current_resolution(self, resolution: int | None = None) -> None:
        """Select the resolution used by progressive diffusion training.

        Convolution kernels are spatially reusable, and decoder activations are
        aligned to their skip tensors dynamically. The selected resolution may
        therefore be lower than, higher than, or equal to ``image_size``.
        ``DiffusionModel`` resizes input batches to this value before calling
        the network.

        Args:
            resolution: Positive square image size. ``None`` restores the
                constructor's native ``image_size``.

        Returns:
            ``None``. The method updates ``current_resolution`` and clears
            cached Keras execution functions when the value changes.
        """

        resolution = self.image_size if resolution is None else resolution
        if not isinstance(resolution, int) or isinstance(resolution, bool):
            raise ValueError("resolution must be an integer.")
        if resolution < 1:
            raise ValueError("resolution must be positive.")

        if getattr(self, "_current_resolution", None) != resolution:
            self._current_resolution = resolution
            self.train_function = None
            self.test_function = None
            self.predict_function = None

    def build_model(
        self,
        call_model: bool = True,
    ) -> list[tf.TensorShape]:
        """Create symbolic inputs and optionally execute the U-Net graph.

        This mirrors ``DiffusionTransformer.build_model`` and gives
        ``DiffusionModel`` an eagerly built, cloneable network for raw/EMA
        weight synchronization.

        Args:
            call_model: Whether to call the U-Net with the symbolic inputs and
                store its symbolic output. Set false only when a caller needs
                input shapes without executing the graph.

        Returns:
            Shapes for noisy images, timesteps, and labels, in that order.
        """

        noisy_images = layers.Input(
            shape=(
                self._current_resolution,
                self._current_resolution,
                self.channels,
            ),
            dtype=self.compute_dtype,
            name="noisy_images",
        )
        timesteps = layers.Input(
            shape=(),
            dtype=tf.int32,
            name="timesteps",
        )
        labels = layers.Input(
            shape=(),
            dtype=tf.int32,
            name="labels",
        )

        self.inputs = (noisy_images, timesteps, labels)
        self.outputs = self.call(self.inputs) if call_model else None

        return [input_tensor.shape for input_tensor in self.inputs]

    def add_depths(self, depth_spec: object) -> NoReturn:
        """Reject transformer-style structural growth for this fixed U-Net.

        Timestep and resolution tasks in ``fit_progressively`` are fully
        supported. Adding encoder/decoder levels after optimizer creation would
        require a separate skip topology and output-head migration policy, so
        this simple U-Net deliberately keeps its architecture fixed.

        Args:
            depth_spec: Depth specification received from
                ``DiffusionModel.fit_progressively``.

        Returns:
            This method never returns; it raises ``NotImplementedError``.
        """

        raise NotImplementedError(
            "UNet does not support progressive depth tasks. Use timestep and "
            "resolution tasks, or construct a deeper UNet through widths, "
            "block_depth, and bottleneck_depth before training."
        )
