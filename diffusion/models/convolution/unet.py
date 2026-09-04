"""Depth-based conditional convolutional U-Net for diffusion training."""

import tensorflow as tf
from tensorflow.keras import layers, models

import math

from collections.abc import Mapping, Sequence

from copy import deepcopy

from . import UNetFullOutput, UNetInputs

from common.argument_saver import ArgumentSaverModel
from common.keras_registry import register_canonical_keras_serializable
from common.runtime import derive_seed

from diffusion.layers.embedding.condition_embedding import ConditionEmbedding
from diffusion.layers.feature_handler import FeatureHandler
from diffusion.layers.convolution import ImageDownsample
from diffusion.layers.convolution import ImageUpsample
from diffusion.layers.convolution import LayerDict
from diffusion.layers.convolution import ResidualConvStack
from diffusion.layers.convolution import VariationalReshaper

@register_canonical_keras_serializable(package="continual_learning")
class UNet(ArgumentSaverModel):
    """Build a hierarchical convolutional diffusion network.

    The model follows the public raw-network contract of
    :class:`DiffusionTransformer`: depth 0 is the embedded image, every later
    depth is stored in ``layers_dicts``, ``encode`` returns intermediate
    features and auxiliary values, and ``call(..., full_return=True)`` returns
    ``(noise, condition, features, regularizers, latent_statistics)``.

    ``widths`` creates the encoder hierarchy. Each encoder level has one
    residual stack and one downsampler. A residual bottleneck follows, then the
    decoder upsamples through the widths in reverse order. Normal U-Net mode
    uses encoder skips. Setting ``reshaper_kwargs={"add_kl": True}`` inserts a
    flatten/unflatten variational bottleneck and disables skips by default, so
    :meth:`DiffusionModel.sample_vae` can decode a latent without bypassing it.
    Explicitly enabling skips inserts a variational pair at every encoder level,
    making each routed skip independently sampleable.

    Args:
        num_classes: Number of real dataset classes, or ``None`` for dynamic
            continual growth. After growth, ``get_config()`` records the
            current class width for checkpoint reconstruction.
        use_cfg: Reserve label ID 0 for classifier-free guidance.
        timesteps: Number of discrete diffusion timesteps.
        image_size: Native square input resolution.
        channels: Input and output image channels.
        widths: Encoder feature widths from high to low resolution.
        block_depth: Residual blocks per encoder and decoder stack.
        bottleneck_width: Bottleneck feature width.
        bottleneck_depth: Residual blocks in the bottleneck stack.
        image_embedding_dim: Width of the initial 1x1 image projection.
        time_embedding_dim: Timestep embedding width.
        label_embedding_dim: Label embedding width.
        activation_func: Keras activation used in residual blocks.
        final_activation_func: Activation applied to predicted noise.
        use_batch_norm: Enable batch normalization in residual blocks.
        dropout_rate: Spatial dropout rate inside residual blocks.
        downsampling_method: ``avg_pooling``, ``max_pooling``, or
            ``cnn_stride``.
        upsampling_method: ``interpolate``, ``cnn_interpolate``, or
            ``cnn_transpose``.
        upsampling_interpolation: Interpolation used by image upsamplers.
        use_skip_connections: Use encoder-to-decoder skips. ``None`` enables
            them for an ordinary U-Net and disables them for a reshaped/VAE
            bottleneck. Explicit ``True`` makes every skip stochastic.
        reshaper_ids_dict: Optional exact model-computed flatten/unflatten
            mapping. Leave it empty to create the required stages automatically.
        reshaper_kwargs: ``add_kl`` and an optional ``latent_dim_ratio`` list
            ordered by ascending flatten/unflatten pair depth.
        cls_token_regularizer_ids: Depth IDs for auxiliary class heads. ID 0
            regularizes the label embedding; ``[None]`` selects every depth.
            The historical name is retained for wrapper compatibility.
        cls_token_regularizer_kwargs: Retained transformer-compatible metadata.
        extra_depth_specs: Shape-preserving residual depths previously added by
            :meth:`add_depths`; normally left empty at construction.
        name_prefix: Prefix for generated layer names.
        build: Build all variables immediately for EMA cloning.
        **kwargs (object): Standard Keras model options.
    """

    FC = "0_feature_connector"
    CB = "1_convolution_block"
    DS = "2_downsampler"
    US = "3_upsampler"
    R = "4_reshaper"
    CTR = "5_cls_token_regularizer"

    def __init__(
        self, 
        num_classes: int | None = 10, 
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
        dropout_rate: float = 0.0, 
        downsampling_method: str = "avg_pooling", 
        upsampling_method: str = "interpolate", 
        upsampling_interpolation: str = "bilinear", 
        use_skip_connections: bool | None = None, 
        reshaper_ids_dict: Mapping[int, str] = {}, 
        reshaper_kwargs: Mapping[str, object] = {}, 
        cls_token_regularizer_ids: Sequence[int | None] = (), 
        cls_token_regularizer_kwargs: Mapping[str, object] = {
            "start": 0, 
            "end": 1, 
            "train_type": "normal",
            "distil_type": "hard",
        }, 
        extra_depth_specs: Sequence[object] = (), 
        name_prefix: str = "", 
        seed: int | None = None,
        build: bool = True, 
        **kwargs: object
    ) -> None:
        """Initialize a conditional convolutional diffusion U-Net.

        Args:
            num_classes (int | None): Positive number of real classes, or
                ``None`` for dynamic continual growth with CFG.
            use_cfg (bool): Whether label ID 0 is reserved for CFG.
            timesteps (int): Positive timestep-embedding vocabulary size.
            image_size (int): Positive native square image side.
            channels (int): Positive input/output channel count.
            widths (Sequence[int]): Positive encoder widths from high to low
                spatial resolution.
            block_depth (int): Positive residual-stack depth per level.
            bottleneck_width (int): Positive bottleneck channel width.
            bottleneck_depth (int): Positive bottleneck residual depth.
            image_embedding_dim (int): Positive image-projection width.
            time_embedding_dim (int): Positive timestep-embedding width.
            label_embedding_dim (int): Positive label-embedding width.
            activation_func (str): Keras residual activation name.
            final_activation_func (str): Keras output activation name.
            use_batch_norm (bool): Enable residual batch normalization.
            dropout_rate (float): Spatial dropout probability in ``[0,1)``.
            downsampling_method (str): ImageDownsample method name.
            upsampling_method (str): ImageUpsample method name.
            upsampling_interpolation (str): TensorFlow resize method.
            use_skip_connections (bool | None): Explicit skip behavior; None
                disables skips only for a variational bottleneck; True creates
                stochastic multiscale skips.
            reshaper_ids_dict (Mapping[int, str]): Optional exact generated
                flatten/unflatten mapping.
            reshaper_kwargs (Mapping[str, object]): ``add_kl`` and an optional
                list of positive ``latent_dim_ratio`` values, one per generated
                flatten/unflatten pair in ascending depth order.
            cls_token_regularizer_ids (Sequence[int | None]): Auxiliary class
                head depths; None expands across all depths.
            cls_token_regularizer_kwargs (Mapping[str, object]): Compatibility
                mapping containing integer ``start`` and ``end`` keys.
                ``train_type`` is ``"normal"``, ``"distil"``, or
                ``"both"``; ``distil_type`` is ``"hard"`` or ``"soft"``.
            extra_depth_specs (Sequence[object]): Serialized progressive stages.
            name_prefix (str): Prefix for generated Keras layer names.
            seed (int | None): Optional raw-network seed used to derive
                independent spatial-dropout streams.
            build (bool): Build variables immediately when true.
            **kwargs (object): Standard ``tf.keras.Model`` options.

        Returns:
            None: The model and optional symbolic graph are initialized in place.
        """

        widths = tuple(widths)
        reshaper_ids_dict = {
            int(key) if isinstance(key, str) and key.lstrip("-").isdigit()
            else key: value
            for key, value in dict(reshaper_ids_dict).items()
        }
        reshaper_kwargs = dict(reshaper_kwargs)
        cls_token_regularizer_ids = list(cls_token_regularizer_ids)
        cls_token_regularizer_kwargs = dict(cls_token_regularizer_kwargs)
        cls_token_regularizer_kwargs.setdefault("train_type", "normal")
        cls_token_regularizer_kwargs.setdefault("distil_type", "hard")
        extra_depth_specs = list(extra_depth_specs)

        derive_seed(seed, "unet", "validation")
        seed = None if seed is None else int(seed)
        super().__init__(**kwargs)
        self._check_arguments(
            num_classes=num_classes, 
            use_cfg=use_cfg,
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
            dropout_rate=dropout_rate, 
            use_skip_connections=use_skip_connections, 
            reshaper_kwargs=reshaper_kwargs, 
            cls_token_regularizer_kwargs=cls_token_regularizer_kwargs, 
        )
        self._save_init_args(locals())
        self._init_config.update({
            "name": self.name, 
            "trainable": self.trainable, 
            "dtype": self.dtype_policy.name, 
            "dynamic": self.dynamic, 
        })

        self.dynamic_num_classes = self.num_classes is None
        self.num_classes = 0 if self.dynamic_num_classes else self.num_classes
        self.num_labels = self.num_classes + int(self.use_cfg)
        self.condition_dim = self.time_embedding_dim + self.label_embedding_dim
        self.use_reshaper = bool(self.reshaper_ids_dict) or bool(
            self.reshaper_kwargs.get("add_kl", False)
        )
        # Default to skip connections unless the resumable VAE path is active.
        if self.use_skip_connections is None:
            self.use_skip_connections = not self.use_reshaper
        self._base_depth = self._compute_base_depth()
        if self.use_reshaper and self.use_skip_connections:
            expected_reshapers = {
                depth: reshape_type
                for level in range(len(self.widths) + 1)
                for depth, reshape_type in (
                    (4 * level + 2, "flatten"),
                    (4 * level + 3, "unflatten"),
                )
            }
        elif self.use_reshaper:
            flatten_depth = 2 * len(self.widths) + 2
            expected_reshapers = {
                flatten_depth: "flatten",
                flatten_depth + 1: "unflatten",
            }
        else:
            expected_reshapers = {}
        if self.reshaper_ids_dict and self.reshaper_ids_dict != expected_reshapers:
            raise ValueError(
                "reshaper_ids_dict must match the model reshaper stages "
                f"{expected_reshapers}."
            )
        self.reshaper_ids_dict = expected_reshapers
        pair_count = sum(
            reshape_type == "flatten"
            for reshape_type in self.reshaper_ids_dict.values()
        )
        latent_dim_ratios = self.reshaper_kwargs.get("latent_dim_ratio")
        if latent_dim_ratios is None or len(latent_dim_ratios) == 0:
            latent_dim_ratios = [1.0] * pair_count
        if len(latent_dim_ratios) != pair_count:
            raise ValueError(
                "latent_dim_ratio must contain one value per "
                "flatten/unflatten pair."
            )
        self.reshaper_kwargs["latent_dim_ratio"] = [
            float(ratio) for ratio in latent_dim_ratios
        ]
        self.depth = self._base_depth + len(self.extra_depth_specs)
        self.cls_token_regularizer_ids = self._handle_ids(
            self.cls_token_regularizer_ids, 
            depth=self.depth, 
            min_id=0, 
            max_id=self.depth, 
        )
        self.connection_ids_dict: dict[int, list[int]] = self._no_dependency({})
        self.cross_attention_ids_dict: dict[int, list[int]] = self._no_dependency({})
        self.set_current_resolution()

        # Persist resolved values so EMA/config cloning recreates one topology.
        self._init_config.update({
            "use_skip_connections": self.use_skip_connections, 
            "reshaper_ids_dict": deepcopy(self.reshaper_ids_dict), 
            "reshaper_kwargs": deepcopy(self.reshaper_kwargs), 
            "cls_token_regularizer_ids": list(self.cls_token_regularizer_ids), 
            "extra_depth_specs": deepcopy(self.extra_depth_specs), 
        })

        self.image_embedder = layers.Conv2D(
            filters=self.image_embedding_dim, 
            kernel_size=1, 
            name=f"{self.name_prefix}image_embedder", 
            dtype=self.dtype_policy, 
        )
        # DiffusionClassifierV2 uses the transformer's historical attribute.
        self.patch_embedder = self.image_embedder
        self.time_embedder = ConditionEmbedding(
            dim=self.time_embedding_dim, 
            pos_embed_type="1d_sincos", 
            embed_steps=self.timesteps, 
            embed_trainable=False, 
            name=f"{self.name_prefix}time_embedder", 
            dtype=self.dtype_policy, 
        )
        self.label_embedder = ConditionEmbedding(
            dim=self.label_embedding_dim, 
            pos_embed_type="new_weight", 
            embed_steps=self.num_labels, 
            embed_trainable=True, 
            name=f"{self.name_prefix}label_embedder", 
            dtype=self.dtype_policy, 
        )
        self.labels_embed_reg = self._create_regularizer(
            f"{self.name_prefix}labels_embed_regularizer", 
            spatial=False,
        ) if 0 in self.cls_token_regularizer_ids else None
        self.cls_token = None

        self._stage_kinds: list[str] = []
        self.layers_dicts: list[LayerDict] = []
        self._create_layers()
        self.output_projection = layers.Conv2D(
            filters=self.channels, 
            kernel_size=1, 
            kernel_initializer="zeros", 
            bias_initializer="zeros", 
            name=f"{self.name_prefix}noise_projection", 
            dtype=self.dtype_policy, 
        )
        self.output_activation = layers.Activation(
            self.final_activation_func, 
            name=f"{self.name_prefix}predicted_noise", 
            dtype=self.dtype_policy, 
        )

        # Materialize symbolic inputs and variables when eager construction is requested.
        if self.build_:
            self.build()

    @staticmethod
    def _check_arguments(
        num_classes: int | None,
        use_cfg: bool,
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
        dropout_rate: float, 
        use_skip_connections: bool | None, 
        reshaper_kwargs: dict, 
        cls_token_regularizer_kwargs: dict
    ) -> None:
        """Validate structural constructor combinations and option names.

        Args:
            num_classes (int | None): Real class count or dynamic sentinel.
            use_cfg (bool): Whether dynamic mode has a null-label row.
            timesteps (int): Discrete timestep count.
            image_size (int): Native square image side.
            channels (int): Image channel count.
            widths (tuple[int, ...]): Encoder widths.
            block_depth (int): Encoder/decoder residual depth.
            bottleneck_width (int): Bottleneck width.
            bottleneck_depth (int): Bottleneck residual depth.
            image_embedding_dim (int): Image embedding width.
            time_embedding_dim (int): Time embedding width.
            label_embedding_dim (int): Label embedding width.
            dropout_rate (float): Dropout probability.
            use_skip_connections (bool | None): Requested skip behavior.
            reshaper_kwargs (dict[str, object]): Variational reshaper options;
                an explicit ``latent_dim_ratio`` is a per-pair list.
            cls_token_regularizer_kwargs (dict[str, object]): Compatibility
                slice and training policy.

        Returns:
            None: Valid inputs return normally; invalid inputs raise ValueError.
        """

        # Dynamic construction needs the CFG null row as its initial vocabulary.
        if num_classes is None and not use_cfg:
            raise ValueError("num_classes=None requires use_cfg=True.")

        unknown_reshaper_keys = set(reshaper_kwargs) - {
            "add_kl", 
            "latent_dim_ratio", 
        }
        # Reject reshaper options outside the documented VAE controls.
        if unknown_reshaper_keys:
            raise ValueError(
                f"Unknown reshaper kwargs: {sorted(unknown_reshaper_keys)}."
            )
        # Require both slice bounds and no unknown regularizer options.
        allowed_regularizer_keys = {
            "start", "end", "train_type", "distil_type"
        }
        # Require the regularizer bounds and reject unsupported metadata keys.
        if not {"start", "end"} <= set(cls_token_regularizer_kwargs) \
        or not set(cls_token_regularizer_kwargs) <= allowed_regularizer_keys:
            raise ValueError(
                "cls_token_regularizer_kwargs must contain start and end and "
                "may contain train_type and distil_type."
            )
        # Restrict regularizer training to the documented target sources.
        if cls_token_regularizer_kwargs["train_type"] not in (
            "normal", "distil", "both"
        ):
            raise ValueError(
                "cls_token_regularizer_kwargs train_type must be normal, "
                "distil, or both."
            )
        # Restrict teacher targets to hard-label or soft-probability training.
        if cls_token_regularizer_kwargs["distil_type"] not in (
            "hard", "soft"
        ):
            raise ValueError(
                "cls_token_regularizer_kwargs distil_type must be hard or soft."
            )

    def _compute_base_depth(self) -> int:
        """Compute the fixed encoder, bottleneck, and decoder stage count.

        Returns:
            int: Number of non-progressive U-Net stages.
        """

        encoder_depth = 2 * len(self.widths) + 2 * len(self.widths) * int(
            self.use_reshaper and self.use_skip_connections
        )
        bottleneck_depth = 1 + 2 * int(self.use_reshaper)
        decoder_depth = len(self.widths) * (
            3 if self.use_skip_connections else 2
        )

        return encoder_depth + bottleneck_depth + decoder_depth

    @staticmethod
    def _handle_ids(
        ids_dict: dict | Sequence[int | None], 
        depth: int | None, 
        min_id: int = 0, 
        max_id: int | None = None
    ) -> dict[int, list[int]] | list[int]:
        """Normalize ``None`` and negative IDs like DiffusionTransformer.

        Args:
            ids_dict (dict | Sequence[int | None]): Mapping or shorthand IDs.
            depth (int | None): Depth used to resolve negative IDs.
            min_id (int): Smallest permitted/expanded ID.
            max_id (int | None): Largest permitted/expanded ID.

        Returns:
            dict[int, list[int]] | list[int]: Normalized shape matching the
            mapping-versus-sequence form of ``ids_dict``.
        """

        is_dict = isinstance(ids_dict, dict)
        values_dict = ids_dict if is_dict else {1: list(ids_dict)}
        values_dict = {key: list(value) for key, value in values_dict.items()}
        for key, values in values_dict.items():
            upper = key if max_id is None else max_id
            # Expand the None sentinel to every valid layer ID.
            if None in values:
                values = list(range(min_id, upper + 1))
            fixed = []
            for value in values:
                # Resolve negative IDs relative to the final depth.
                if value < 0:
                    # Negative IDs need a known depth for normalization.
                    if depth is None:
                        raise ValueError("Negative IDs require a known depth.")
                    value += depth + 1
                # Reject normalized IDs outside the allowed inclusive range.
                if value < min_id or (max_id is not None and value > max_id):
                    raise ValueError(
                        f"Layer ID {value} is outside [{min_id}, {upper}]."
                    )
                fixed.append(value)
            values_dict[key] = fixed

        return values_dict if is_dict else values_dict[1]

    def _create_regularizer(
        self, 
        name: str, 
        spatial: bool, 
    ) -> models.Model:
        """Create an auxiliary head in the policy's stable variable dtype.

        Args:
            name (str): Keras model/layer name prefix.
            spatial (bool): Pool rank-four feature maps before classification.

        Returns:
            tf.keras.Model: Optional global pool followed by class softmax.
        """

        regularizer_layers = []
        # Pool spatial feature maps before the regularizer classifier.
        if spatial:
            regularizer_layers.append(
                layers.GlobalAveragePooling2D(
                    dtype=self.dtype_policy,
                    name=f"{name}/pool",
                )
            )

        return models.Sequential(
            [
                *regularizer_layers,
                layers.Dense(
                    self.num_classes,
                    activation="softmax",
                    dtype=self.dtype_policy.variable_dtype,
                    name=f"{name}/classes",
                ),
            ],
            name=name,
        )

    def _expand_regularizer(
        self,
        regularizer: layers.Layer | None,
        source_regularizer: layers.Layer | None = None,
    ) -> layers.Layer | None:
        """Append one softmax output while preserving existing head weights.

        Args:
            regularizer (tf.keras.layers.Layer | None): Head to expand.
            source_regularizer (tf.keras.layers.Layer | None): Optional
                expanded raw head supplying the new EMA output parameters.

        Returns:
            tf.keras.layers.Layer | None: Expanded head, or ``None`` when the
            input head is disabled.
        """

        # Leave disabled auxiliary heads untouched.
        if regularizer is None:
            return None

        old_layer = regularizer.layers[-1] \
            if isinstance(regularizer, models.Sequential) else regularizer
        old_kernel, old_bias = old_layer.get_weights()
        layer_config = old_layer.get_config()
        layer_config["units"] = self.num_classes
        new_layer = old_layer.__class__.from_config(layer_config)
        new_layer(
            tf.zeros((1, old_kernel.shape[0]), dtype=old_kernel.dtype),
            training=False,
        )
        new_kernel, new_bias = new_layer.get_weights()
        new_kernel[..., :-1] = old_kernel
        new_bias[:-1] = old_bias

        # Initialize only the new EMA output from the expanded raw head.
        if source_regularizer is not None:
            source_layer = source_regularizer.layers[-1] \
                if isinstance(source_regularizer, models.Sequential) \
                else source_regularizer
            source_kernel, source_bias = source_layer.get_weights()
            new_kernel[..., -1] = source_kernel[..., -1]
            new_bias[-1] = source_bias[-1]

        new_layer.set_weights([new_kernel, new_bias])
        # Retain pooling or hidden layers and replace only the final softmax.
        if isinstance(regularizer, models.Sequential):
            regularizer.pop()
            regularizer.add(new_layer)
            return regularizer

        return new_layer

    def _append_stage(
        self, 
        layers_dict: dict[str, layers.Layer], 
        kind: str
    ) -> int:
        """Track one ordered stage and attach its configured regularizer.

        Args:
            layers_dict (dict[str, tf.keras.layers.Layer]): Stage components.
            kind (str): Internal representation/stage kind.

        Returns:
            int: One-based public depth assigned to the appended stage.
        """

        key = len(self.layers_dicts) + 1
        # Attach a requested regularizer unless this stage already owns one.
        if key in self.cls_token_regularizer_ids and self.CTR not in layers_dict:
            layers_dict[self.CTR] = self._create_regularizer(
                f"{self.name_prefix}depth_{key}_{self.CTR[2:]}",
                spatial=kind != "flatten",
            )
        stage = LayerDict(
            layers_dict, 
            name=f"{self.name_prefix}depth_{key}", 
            dtype=self.dtype_policy, 
        )
        self.layers_dicts.append(stage)
        self._stage_kinds.append(kind)

        return key

    def _residual_stack(
        self, 
        filters: int, 
        depth: int, 
        name: str
    ) -> ResidualConvStack:
        """Create a condition-aware residual convolution stack.

        Args:
            filters (int): Positive output channel width.
            depth (int): Positive number of residual blocks.
            name (str): Keras layer name.

        Returns:
            ResidualConvStack: Configured residual stack.
        """

        return ResidualConvStack(
            filters=filters, 
            depth=depth, 
            condition_dim=self.condition_dim, 
            activation_func=self.activation_func, 
            use_batch_norm=self.use_batch_norm, 
            dropout_rate=self.dropout_rate, 
            seed=derive_seed(self.seed, "residual_stack", name),
            name=name, 
            dtype=self.dtype_policy, 
        )

    def _bottleneck_resolution(self, resolution: int) -> int:
        """Compute the spatial side after every encoder downsample.

        Args:
            resolution (int): Positive input image side.

        Returns:
            int: Repeated ceiling-halved bottleneck side.
        """

        for _ in self.widths:
            resolution = (resolution + 1) // 2
        return resolution

    def _create_layers(self) -> None:
        """Construct encoder, bottleneck, decoder, and progressive stages.

        Returns:
            None: ``layers_dicts`` and routing metadata are populated in place.
        """

        skip_depths = []
        flatten_index = 0
        base_side = self.image_size
        for width in self.widths:
            block_key = len(self.layers_dicts) + 1
            self._append_stage(
                {
                    self.CB: self._residual_stack(
                        width, 
                        self.block_depth, 
                        f"{self.name_prefix}depth_{block_key}_{self.CB[2:]}", 
                    )
                }, 
                "convolution", 
            )
            if self.use_reshaper and self.use_skip_connections:
                source_shape = (base_side, base_side, width)
                flatten_key = len(self.layers_dicts) + 1
                flatten_name = (
                    f"{self.name_prefix}depth_{flatten_key}_{self.R[2:]}"
                )
                self._append_stage(
                    {
                        self.R: VariationalReshaper(
                            "flatten",
                            source_shape,
                            add_kl=bool(
                                self.reshaper_kwargs.get("add_kl", False)
                            ),
                            latent_dim_ratio=self.reshaper_kwargs[
                                "latent_dim_ratio"
                            ][flatten_index],
                            seed=derive_seed(
                                self.seed, "reshaper", flatten_name
                            ),
                            name=flatten_name,
                            dtype=self.dtype_policy,
                        )
                    },
                    "flatten",
                )
                flatten_index += 1
                unflatten_key = len(self.layers_dicts) + 1
                self._append_stage(
                    {
                        self.R: VariationalReshaper(
                            "unflatten",
                            source_shape,
                            name=(
                                f"{self.name_prefix}depth_{unflatten_key}_"
                                f"{self.R[2:]}"
                            ),
                            dtype=self.dtype_policy,
                        )
                    },
                    "unflatten",
                )
                skip_depths.append(unflatten_key)
            else:
                skip_depths.append(block_key)
            down_key = len(self.layers_dicts) + 1
            self._append_stage(
                {
                    self.DS: ImageDownsample(
                        filters=width, 
                        scaling_method=self.downsampling_method, 
                        name=f"{self.name_prefix}depth_{down_key}_{self.DS[2:]}", 
                        dtype=self.dtype_policy, 
                    )
                },
                "downsample",
            )
            base_side = (base_side + 1) // 2

        bottleneck_key = len(self.layers_dicts) + 1
        self._append_stage(
            {
                self.CB: self._residual_stack(
                    self.bottleneck_width, 
                    self.bottleneck_depth, 
                    f"{self.name_prefix}depth_{bottleneck_key}_{self.CB[2:]}", 
                )
            }, 
            "convolution",
        )

        # Build the output head from bottleneck geometry in resumable VAE mode.
        if self.use_reshaper:
            source_shape = (base_side, base_side, self.bottleneck_width)
            flatten_key = len(self.layers_dicts) + 1
            flatten_name = f"{self.name_prefix}depth_{flatten_key}_{self.R[2:]}"
            self._append_stage(
                {
                    self.R: VariationalReshaper(
                        "flatten",  
                        source_shape, 
                        add_kl=bool(self.reshaper_kwargs.get("add_kl", False)), 
                        latent_dim_ratio=self.reshaper_kwargs[
                            "latent_dim_ratio"
                        ][flatten_index],
                        seed=derive_seed(self.seed, "reshaper", flatten_name),
                        name=flatten_name, 
                        dtype=self.dtype_policy, 
                    )
                }, 
                "flatten", 
            )
            flatten_index += 1
            unflatten_key = len(self.layers_dicts) + 1
            self._append_stage(
                {
                    self.R: VariationalReshaper(
                        "unflatten", 
                        source_shape, 
                        name=(
                            f"{self.name_prefix}depth_{unflatten_key}_"
                            f"{self.R[2:]}"
                        ), 
                        dtype=self.dtype_policy, 
                    )
                }, 
                "unflatten", 
            )

        for width, skip_depth in zip(reversed(self.widths), reversed(skip_depths)):
            up_key = len(self.layers_dicts) + 1
            self._append_stage(
                {
                    self.US: ImageUpsample(
                        filters=width, 
                        scaling_method=self.upsampling_method, 
                        interpolation=self.upsampling_interpolation, 
                        name=f"{self.name_prefix}depth_{up_key}_{self.US[2:]}", 
                        dtype=self.dtype_policy, 
                    )
                }, 
                "upsample", 
            )

            # Route the matching encoder feature into each decoder stage.
            if self.use_skip_connections:
                connector_key = len(self.layers_dicts) + 1
                self.connection_ids_dict[connector_key] = [skip_depth]
                self._append_stage(
                    {
                        self.FC: FeatureHandler(
                            ids=[skip_depth], 
                            connect_axis=-1, 
                            connect_type="concat", 
                            use_layer_norm=False, 
                            ln_dim=2 * width, 
                            name=(
                                f"{self.name_prefix}depth_{connector_key}_"
                                f"{self.FC[2:]}"
                            ), 
                            dtype=self.dtype_policy, 
                        )
                    }, 
                    "connector", 
                )
            block_key = len(self.layers_dicts) + 1
            self._append_stage(
                {
                    self.CB: self._residual_stack(
                        width, 
                        self.block_depth, 
                        f"{self.name_prefix}depth_{block_key}_{self.CB[2:]}", 
                    )
                }, 
                "convolution",
            )

        for spec in self.extra_depth_specs:
            self._append_extra_stage(spec)

        # Guard against construction metadata and executable stages diverging.
        if len(self.layers_dicts) != self.depth:
            raise RuntimeError("Constructed U-Net depth does not match its config.")

    def _normalize_extra_spec(self, spec: object) -> dict[str, object]:
        """Normalize one progressive stage specification.

        Args:
            spec (object): Layer name, collection of names, or option mapping.

        Returns:
            dict[str, object]: Canonical component-key mapping.

        Raises:
            ValueError: If no supported enabled component is specified.
        """

        # Interpret a string as one enabled progressive layer.
        if isinstance(spec, str):
            spec = {spec: True}
        # Interpret a collection as several enabled layers in one depth.
        elif isinstance(spec, (tuple, set, frozenset)):
            spec = dict.fromkeys(spec, True)
        aliases = {
            "convolution_block": self.CB, 
            "residual_block": self.CB, 
            "vision_transformer_block": self.CB, 
            self.CB: self.CB, 
            "cls_token_regularizer": self.CTR, 
            self.CTR: self.CTR, 
        }

        normalized = {}
        for name, options in spec.items():
            # Reject progressive layer names unsupported by the U-Net factory.
            if name not in aliases:
                raise ValueError(f"Unknown progressive U-Net layer: {name}.")
            # Retain enabled layers under their internal stage keys.
            if options is not False:
                normalized[aliases[name]] = options

        # Reject a progressive stage with every layer disabled.
        if not normalized:
            raise ValueError("A progressive depth must contain a layer.")

        return normalized

    def _append_extra_stage(self, spec: object) -> int:
        """Build and append one normalized progressive stage.

        Args:
            spec (object): Progressive stage specification.

        Returns:
            int: One-based depth assigned to the new stage.
        """

        spec = self._normalize_extra_spec(spec)
        key = len(self.layers_dicts) + 1
        stage_layers = {}

        # Add a convolutional residual block when requested.
        if self.CB in spec:
            stage_layers[self.CB] = self._residual_stack(
                self.widths[0], 
                self.block_depth, 
                f"{self.name_prefix}depth_{key}_{self.CB[2:]}", 
            )

        # Add an auxiliary classifier regularizer when requested.
        if self.CTR in spec:
            stage_layers[self.CTR] = self._create_regularizer(
                f"{self.name_prefix}depth_{key}_{self.CTR[2:]}", 
                spatial=True,
            )

        return self._append_stage(stage_layers, "extra")

    def _broadcast_condition(
        self, 
        condition: tf.Tensor, 
        images: tf.Tensor
    ) -> tf.Tensor:
        """Broadcast rank-two conditions across an image grid.

        Args:
            condition (tf.Tensor): Float condition tensor ``[B, condition_dim]``.
            images (tf.Tensor): Reference image tensor ``[B,H,W,C]``.

        Returns:
            tf.Tensor: Condition tensor ``[B,H,W,condition_dim]`` in the image
            dtype.
        """

        condition = tf.cast(condition, images.dtype)
        condition = condition[:, None, None, :]
        shape = tf.concat(
            [tf.shape(images)[:3], [self.condition_dim]], 
            axis=0, 
        )
        condition = tf.broadcast_to(condition, shape)
        condition.set_shape(
            (None, None, None, self.condition_dim)
        )

        return condition

    def embed_conditions(
        self, 
        times: tf.Tensor, 
        labels: tf.Tensor, 
        full_return: bool = False, 
        training: bool | None = None
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Embed and concatenate timestep and label conditions.

        Args:
            times (tf.Tensor): Integer timestep IDs of shape ``[B]``.
            labels (tf.Tensor): Integer label IDs of shape ``[B]``.
            full_return (bool): Include the two individual embeddings.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor | tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Concatenated
            condition ``[B, condition_dim]``, optionally followed by individual
            time and label embeddings.
        """
        times = tf.convert_to_tensor(times)
        labels = tf.convert_to_tensor(labels)
        time_embeddings = self.time_embedder(times, training=training)
        label_embeddings = self.label_embedder(labels, training=training)
        condition = tf.concat([time_embeddings, label_embeddings], axis=-1)
        condition = tf.cast(condition, self.compute_dtype)

        # Expose separate embeddings only for callers requesting full metadata.
        if full_return:
            return condition, time_embeddings, label_embeddings
        return condition

    def encode(
        self, 
        inputs: UNetInputs, 
        max_depth: int = -1, 
        training: bool | None = None, 
        min_depth: int = 0
    ) -> tuple[
        tf.Tensor, 
        tf.Tensor, 
        list[tf.Tensor | None], 
        list[tf.Tensor | None], 
        list[tuple[tf.Tensor, tf.Tensor]],
    ]:
        """Run a contiguous range of convolutional depths.

        Args:
            inputs (UNetInputs): Image/latent tensor, integer timesteps, and
                integer labels.  At depth zero the first tensor is ``[B,H,W,C]``;
                at a resumed depth it is that depth's representation.
            max_depth (int): Exclusive zero-based stage stop; -1 runs all.
            training (bool | None): Keras training mode.
            min_depth (int): First stage to execute in ``[0, depth]``.

        Returns:
            tuple[tf.Tensor, tf.Tensor, list[tf.Tensor | None],
            list[tf.Tensor | None], list[tuple[tf.Tensor, tf.Tensor]]]:
            Final representation, condition, depth features, auxiliary class
            predictions, and ordered latent mean/log-variance pairs.
        """

        # Require resumed execution to start at a valid stage boundary.
        if not 0 <= min_depth <= self.depth:
            raise ValueError("min_depth must be in [0, depth].")

        condition, _, label_embeddings = self.embed_conditions(
            inputs[1], 
            inputs[2], 
            full_return=True, 
            training=training, 
        )
        # Embed raw images and conditions when executing from the input stage.
        if min_depth == 0:
            x = self.image_embedder(inputs[0], training=training)
            x = tf.concat([x, self._broadcast_condition(condition, x)], axis=-1)
        # Treat the supplied tensor as an already embedded intermediate feature.
        else:
            latent_inputs = list(inputs[0]) if isinstance(
                inputs[0], (list, tuple)
            ) else [inputs[0]]
            expected_latents = 1 + sum(
                flatten_id > min_depth and (
                    max_depth < 0 or flatten_id <= max_depth
                )
                for flatten_id, reshape_type in self.reshaper_ids_dict.items()
                if reshape_type == "flatten"
            )
            # Require one resumed value for every flatten stage still ahead.
            if len(latent_inputs) != expected_latents:
                raise ValueError(
                    f"Resuming at depth {min_depth} requires "
                    f"{expected_latents} input feature/latent tensors."
                )
            x = latent_inputs[0]

        label_reg = self.labels_embed_reg(
            label_embeddings,
            training=training,
        ) if self.labels_embed_reg is not None else None

        features_list: list[tf.Tensor | None] = [None] * min_depth + [x]
        regs_list: list[tf.Tensor | None] = [label_reg] + [None] * min_depth
        z_vals_list: list[tuple[tf.Tensor, tf.Tensor]] = []
        latent_index = 1

        for index, stage in enumerate(self.layers_dicts):
            # Stop before the exclusive maximum depth.
            if index == max_depth:
                break
            # Leave earlier stages untouched during resumed execution.
            if index < min_depth:
                continue

            # Merge the configured encoder skip feature into this decoder stage.
            if self.FC in stage:
                source_id = self.connection_ids_dict[index + 1][0]
                source = features_list[source_id]
                x = tf.image.resize(x, tf.shape(source)[1:3])
                x = tf.cast(x, source.dtype)
                x = stage[self.FC](
                    features_list, 
                    second_list=[x], 
                    training=training, 
                )

            # Upsample decoder features at this stage.
            if self.US in stage:
                x = stage[self.US]((x, condition), training=training)

            # Apply this stage's convolutional residual stack.
            if self.CB in stage:
                x = stage[self.CB]((x, condition), training=training)

            # Downsample encoder features at this stage.
            if self.DS in stage:
                x = stage[self.DS]((x, condition), training=training)

            # Apply the configured flatten or unflatten bottleneck transform.
            if self.R in stage:
                reshape_type = self.reshaper_ids_dict[index + 1]
                if min_depth > 0 and reshape_type == "flatten":
                    x = latent_inputs[latent_index]
                    latent_index += 1
                    x_mean, x_log_var = None, None
                else:
                    if reshape_type == "flatten":
                        x = tf.image.resize(
                            x, stage[self.R].source_shape_[:2]
                        )
                    x, x_mean, x_log_var = stage[self.R](
                        x, training=training
                    )

                # Restore the active geometry at this hierarchy level.
                if reshape_type == "unflatten":
                    side = self.current_resolution
                    for previous_stage in self.layers_dicts[:index]:
                        if self.DS in previous_stage:
                            side = (side + 1) // 2
                    x = tf.image.resize(x, (side, side))

                # Preserve mean and log variance from a variational flatten stage.
                if reshape_type == "flatten" and bool(
                    self.reshaper_kwargs.get("add_kl", False)
                ) and x_mean is not None:
                    z_vals_list.append((x_mean, x_log_var))

            regularizer = stage[self.CTR](
                x, 
                training=training,
            ) if self.CTR in stage else None

            features_list.append(x)
            regs_list.append(regularizer)

        return x, condition, features_list, regs_list, z_vals_list

    def call(
        self, 
        inputs: UNetInputs, 
        full_return: bool = False, 
        training: bool | None = None, 
        min_depth: int = 0
    ) -> tf.Tensor | UNetFullOutput:
        """Predict image-shaped noise and optionally return intermediates.

        Args:
            inputs (UNetInputs): Image/latent, timestep, and label tensors.
            full_return (bool): Return the five-item wrapper contract when true.
            training (bool | None): Keras training mode.
            min_depth (int): Resume execution from this architectural depth.

        Returns:
            tf.Tensor | UNetFullOutput: Predicted noise ``[B,H,W,C]`` or the
            prediction plus condition, features, regularizers, and latent stats.
        """

        x, condition, features_list, regs_list, z_vals_list = self.encode(
            inputs,
            min_depth=min_depth,
            training=training,
        )
        target_size = tf.shape(inputs[0])[1:3] if min_depth == 0 else tf.constant(
            [self.current_resolution, self.current_resolution],
            dtype=tf.int32,
        )
        x = tf.image.resize(x, target_size)
        x = self.output_projection(x, training=training)
        predicted_noise = self.output_activation(x)

        # Return intermediate features and auxiliary values only when requested.
        if full_return:
            return predicted_noise, condition, features_list, regs_list, z_vals_list
        return predicted_noise

    def predict_noise(
        self, 
        inputs: UNetInputs, 
        full_return: bool = False, 
        training: bool | None = None
    ) -> tf.Tensor | UNetFullOutput:
        """Run only the U-Net noise-prediction branch.

        Args:
            inputs (UNetInputs): Image, timestep, and label tensors.
            full_return (bool): Include branch intermediates when true.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor | UNetFullOutput: Same contract as :meth:`call`.
        """

        return UNet.call(
            self, 
            inputs, 
            full_return=full_return, 
            training=training, 
        )

    @property
    def current_resolution(self) -> int:
        """Return the active square image side.

        Returns:
            int: Positive active resolution.
        """

        return self._current_resolution

    def set_current_resolution(self, resolution: int | None = None) -> None:
        """Set the active square resolution and invalidate Keras functions.

        Args:
            resolution (int | None): Positive image side; None restores native.

        Returns:
            None: Resolution and execution caches are updated in place.
        """

        resolution = self.image_size if resolution is None else resolution
        # Invalidate traced functions only when the active resolution changes.
        if getattr(self, "_current_resolution", None) != resolution:
            self._current_resolution = resolution
            self.train_function = None
            self.test_function = None
            self.predict_function = None

    def build(
        self, 
        input_shape: tuple[tf.TensorShape, tf.TensorShape, tf.TensorShape]
        | None = None
    ) -> None:
        """Build variables from model-configured symbolic shapes.

        Args:
            input_shape (tuple[tf.TensorShape, tf.TensorShape,
                tf.TensorShape] | None): Accepted for Keras compatibility and
                ignored in favor of configured shapes.

        Returns:
            None: Variables and the Keras built flag are initialized.
        """

        del input_shape
        shapes = self.build_model()
        super().build(shapes)

    def build_model(self, call_model: bool = True) -> list[tf.TensorShape]:
        """Create symbolic image, timestep, and label inputs.

        Args:
            call_model (bool): Populate symbolic outputs when true.

        Returns:
            list[tf.TensorShape]: Shapes of the three symbolic inputs.
        """

        noisy_images = layers.Input(
            shape=(self.current_resolution, self.current_resolution, self.channels), 
            dtype=self.compute_dtype, 
            name="noisy_images", 
        )
        times = layers.Input(shape=(), dtype=tf.int32, name="timesteps")
        labels = layers.Input(shape=(), dtype=tf.uint8, name="labels")
        self.inputs = (noisy_images, times, labels)
        self.outputs = self.call(self.inputs) if call_model else None

        return [value.shape for value in self.inputs]

    def add_class(self, source_network: object | None = None) -> None:
        """Append one label embedding while preserving existing rows.

        Args:
            source_network (object | None): Optional already-expanded raw
                network whose new embedding row initializes an EMA clone.

        Returns:
            None: The label vocabulary grows by one in place and
            ``get_config()`` records the grown class width.

        Raises:
            ValueError: If the network was initialized with a fixed class count.
        """

        # Keep fixed-width construction on its established immutable path.
        if not self.dynamic_num_classes:
            raise ValueError("add_class requires num_classes=None at initialization.")

        old_label_embedder = self.label_embedder
        old_weights = old_label_embedder.get_weights()
        self.num_classes += 1
        self._init_config["num_classes"] = self.num_classes
        self.num_labels = self.num_classes + int(self.use_cfg)

        label_config = old_label_embedder.get_config()
        label_config["embed_steps"] = self.num_labels
        new_label_embedder = old_label_embedder.__class__.from_config(
            label_config
        )
        new_label_embedder(tf.zeros((1,), dtype=tf.int32), training=False)
        new_weights = new_label_embedder.get_weights()
        new_weights[0][:-1] = old_weights[0]
        for index in range(1, len(old_weights)):
            new_weights[index] = old_weights[index]

        # Initialize the new EMA row from the already-expanded raw network.
        if source_network is not None:
            new_weights[0][-1] = source_network.label_embedder.get_weights()[0][-1]

        new_label_embedder.set_weights(new_weights)
        self.label_embedder = new_label_embedder

        self.labels_embed_reg = self._expand_regularizer(
            self.labels_embed_reg,
            source_network.labels_embed_reg if source_network is not None else None,
        )
        for index, stage in enumerate(self.layers_dicts):
            # Expand only stages that own an auxiliary class head.
            if self.CTR in stage:
                stage[self.CTR] = self._expand_regularizer(
                    stage[self.CTR],
                    source_network.layers_dicts[index][self.CTR]
                    if source_network is not None else None,
                )

    def add_depths(self, depth_spec: object) -> dict[str, dict[str, int]]:
        """Append shape-preserving convolution or regularizer stages.

        Args:
            depth_spec (object): One stage specification or a list of them.

        Returns:
            dict[str, dict[str, int]]: Before, added, and after depth counts.
        """

        specs = depth_spec if isinstance(depth_spec, list) else [depth_spec]
        specs = [spec for spec in specs if spec is not None]
        before = self.depth
        # Leave the architecture unchanged for an empty growth request.
        if not specs:
            return {"network": {"before": before, "added": 0, "after": before}}

        normalized = [self._normalize_extra_spec(spec) for spec in specs]
        serializable_specs = []
        for spec in normalized:
            saved = {}
            # Serialize requested convolution-block options.
            if self.CB in spec:
                saved["convolution_block"] = spec[self.CB]
            # Serialize requested regularizer options.
            if self.CTR in spec:
                saved["cls_token_regularizer"] = spec[self.CTR]
            serializable_specs.append(saved)
            new_depth = self._append_extra_stage(saved)
            # Register newly added regularizer depths in model metadata.
            if self.CTR in spec:
                self.cls_token_regularizer_ids.append(new_depth)

        self.extra_depth_specs.extend(serializable_specs)
        self.depth = len(self.layers_dicts)
        self._init_config["extra_depth_specs"] = deepcopy(self.extra_depth_specs)
        self._init_config["cls_token_regularizer_ids"] = list(
            self.cls_token_regularizer_ids
        )
        self.train_function = None
        self.test_function = None
        self.predict_function = None

        return {
            "network": {
                "before": before, 
                "added": self.depth - before, 
                "after": self.depth, 
            }
        }

    def get_variables_names(
        self, 
        vars: list[tf.Variable] | None = None
    ) -> list[str]:
        """Return TensorFlow names for all or selected trainable variables.

        Args:
            vars (list[tf.Variable] | None): Variables to inspect; None selects
                every trainable variable.

        Returns:
            list[str]: Variable names in model/input order.
        """

        vars = self.trainable_variables if vars is None else vars

        return [variable.name for variable in vars]


# TensorFlow 2.10 writes the plain root name for subclassed model JSON.
tf.keras.utils.get_custom_objects()["UNet"] = UNet


def run_self_tests() -> dict[str, str]:
    """Run direct, wrapper, VAE, progressive, and serialization checks.

    Returns:
        dict[str, str]: ``{"UNet": "passed"}`` after all checks.
    """

    tf.keras.backend.clear_session()
    tf.random.set_seed(211)
    common = {
        "num_classes": 2, 
        "timesteps": 4, 
        "image_size": 5, 
        "channels": 1, 
        "widths": (2, 3), 
        "block_depth": 1, 
        "bottleneck_width": 4, 
        "bottleneck_depth": 1, 
        "image_embedding_dim": 2, 
        "time_embedding_dim": 2, 
        "label_embedding_dim": 2, 
        "use_batch_norm": False, 
        "build": False, 
    }
    model = UNet(**common)
    assert model.reshaper_kwargs["latent_dim_ratio"] == []
    assert model.cls_token_regularizer_kwargs["train_type"] == "normal"
    assert model.cls_token_regularizer_kwargs["distil_type"] == "hard"
    distil_regularized = UNet(
        **common,
        cls_token_regularizer_kwargs={
            "start": 0,
            "end": 1,
            "train_type": "distil",
            "distil_type": "soft",
        },
    )
    assert distil_regularized.cls_token_regularizer_kwargs["train_type"] == "distil"
    assert distil_regularized.cls_token_regularizer_kwargs["distil_type"] == "soft"
    images = tf.ones((2, 5, 7, 1))
    times = tf.constant([0, 3], tf.int32)
    labels = tf.constant([0, 1], tf.uint8)
    output, condition, features, regs, z_vals_list = model(
        (images, times, labels), full_return=True
    )
    assert output.shape == images.shape
    assert condition.shape == (2, 4)
    assert len(features) == len(regs) == model.depth + 1
    assert z_vals_list == []
    assert len(model.layers_dicts) == model.depth
    assert model.connection_ids_dict

    clone = UNet.from_config(model.get_config())
    assert clone((images, times, labels)).shape == images.shape
    assert len(clone.weights) == len(model.weights)
    model.train_function = model.test_function = model.predict_function = object()
    growth = model.add_depths("convolution_block")
    assert growth["network"]["added"] == 1
    assert model.train_function is model.test_function is model.predict_function is None
    assert model((images, times, labels)).shape == images.shape

    for invalid_reshaper_kwargs in (
        {"add_kl": True, "latent_dim_ratio": [0.0]},
    ):
        try:
            UNet(**common, reshaper_kwargs=invalid_reshaper_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Invalid model-level latent_dim_ratio configuration must fail."
            )

    vae = UNet(
        **common, 
        reshaper_kwargs={"add_kl": True},
    )
    assert vae.reshaper_kwargs["latent_dim_ratio"] == [1.0]
    empty_ratio_vae = UNet(
        **common,
        reshaper_kwargs={"add_kl": True, "latent_dim_ratio": []},
    )
    assert empty_ratio_vae.reshaper_kwargs["latent_dim_ratio"] == [1.0]
    square_images = images[:, :5, :5]
    vae_output = vae((square_images, times, labels), full_return=True)
    assert vae_output[0].shape == (2, 5, 5, 1)
    assert vae_output[-1][0][0].shape.rank == 2
    assert vae_output[-1][0][1].shape == vae_output[-1][0][0].shape
    assert not vae.connection_ids_dict and vae.reshaper_ids_dict
    json_clone = tf.keras.models.model_from_json(vae.to_json())
    assert json_clone.reshaper_ids_dict == vae.reshaper_ids_dict
    assert json_clone((square_images, times, labels)).shape == square_images.shape


    from diffusion.models.wrapper.diffusion_model import DiffusionModel


    multiscale_vae = UNet(
        **common,
        use_skip_connections=True,
        reshaper_kwargs={
            "add_kl": True,
            "latent_dim_ratio": [0.5, 1.0, 0.25],
        },
    )
    multiscale_output = multiscale_vae(
        (square_images, times, labels), full_return=True, training=False
    )
    flatten_ids = sorted(
        depth for depth, reshape_type in
        multiscale_vae.reshaper_ids_dict.items()
        if reshape_type == "flatten"
    )
    unflatten_ids = {
        depth for depth, reshape_type in
        multiscale_vae.reshaper_ids_dict.items()
        if reshape_type == "unflatten"
    }
    assert len(multiscale_output[-1]) == len(flatten_ids) == 3
    latent_widths = [
        int(multiscale_vae.layers_dicts[depth - 1][
            multiscale_vae.R
        ].output_shape[1][-1])
        for depth in flatten_ids
    ]
    assert latent_widths == [25, 27, 4]
    assert [
        int(z_mean.shape[-1]) for z_mean, _ in multiscale_output[-1]
    ] == latent_widths
    first_flatten = flatten_ids[0]
    first_source_shape = multiscale_vae.layers_dicts[
        first_flatten - 1
    ][multiscale_vae.R].source_shape_
    truncated_multiscale, *_ = multiscale_vae.encode(
        (
            [tf.zeros((2, math.prod(first_source_shape)))],
            times,
            labels,
        ),
        min_depth=first_flatten,
        max_depth=first_flatten + 1,
        training=False,
    )
    assert truncated_multiscale.shape.rank == 4
    multiscale_clone = tf.keras.models.model_from_json(
        multiscale_vae.to_json()
    )
    assert multiscale_clone.reshaper_kwargs["latent_dim_ratio"] == [
        0.5, 1.0, 0.25
    ]
    assert [
        int(multiscale_clone.layers_dicts[depth - 1][
            multiscale_clone.R
        ].output_shape[1][-1])
        for depth in flatten_ids
    ] == latent_widths
    assert all(
        sources[0] in unflatten_ids
        for sources in multiscale_vae.connection_ids_dict.values()
    )
    kernel, bias = multiscale_vae.output_projection.get_weights()
    multiscale_vae.output_projection.set_weights([
        tf.ones_like(kernel), bias
    ])
    multiscale_wrapper = DiffusionModel(
        network=multiscale_vae,
        use_ema=False,
        test_network_name="raw",
        test_steps=2,
    )
    z_inputs = [
        tf.zeros((1, int(
            multiscale_vae.layers_dicts[depth - 1][
                multiscale_vae.R
            ].output_shape[1][-1]
        )))
        for depth in flatten_ids
    ]
    base_sample = multiscale_wrapper.sample_vae(
        network_name="raw", labels=[1], z=z_inputs
    )
    for latent_index in range(len(z_inputs)):
        changed_inputs = [tf.identity(value) for value in z_inputs]
        changed_inputs[latent_index] = tf.ones_like(
            changed_inputs[latent_index]
        )
        changed_sample = multiscale_wrapper.sample_vae(
            network_name="raw", labels=[1], z=changed_inputs
        )
        assert float(tf.reduce_max(tf.abs(changed_sample - base_sample))) > 0.


    wrapper = DiffusionModel(
        network=vae, 
        use_ema=True, 
        test_network_name="raw", 
        scheduler_name="linear", 
        test_steps=2, 
        p_uncond=0.0, 
        swap_noise_image=True, 
        kl_loss_coef=0.1, 
    )
    wrapper.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3), 
        loss="mse", 
        run_eagerly=True, 
    )
    batch = (square_images, tf.constant([0, 1], tf.uint8))
    results = wrapper.train_step(batch)
    assert "noise_loss" in results and "kl_loss" in results
    samples = wrapper.sample_vae(network_name="raw", labels=[0, 1], seed=5)
    assert samples.shape == (2, 5, 5, 1)

    tf.keras.backend.clear_session()
    return {"UNet": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
