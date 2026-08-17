"""Depth-based conditional convolutional U-Net for diffusion training."""

import tensorflow as tf
from tensorflow.keras import layers, models

from collections.abc import Mapping, Sequence

from copy import deepcopy

from . import UNetFullOutput, UNetInputs

from common.argument_saver import ArgumentSaverModel

from diffusion.layers.embedding.condition_embedding import ConditionEmbedding
from diffusion.layers.feature_handler import FeatureHandler
from diffusion.layers.convolution import ImageDownsample
from diffusion.layers.convolution import ImageUpsample
from diffusion.layers.convolution import LayerDict
from diffusion.layers.convolution import ResidualConvStack
from diffusion.layers.convolution import VariationalReshaper


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

    Args:
        num_classes: Number of real dataset classes.
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
            bottleneck. Explicit ``True`` is rejected with a bottleneck because
            the unchanged VAE sampler cannot provide pre-latent features.
        reshaper_ids_dict: Optional explicit bottleneck depth mapping. The only
            valid mapping is the model-computed consecutive flatten/unflatten
            pair. Leave empty and set ``add_kl`` to create that pair
            automatically.
        reshaper_kwargs: ``add_kl`` and ``latent_dim_ratio``.
        cls_token_regularizer_ids: Depth IDs for auxiliary class heads. ID 0
            regularizes the label embedding; ``[None]`` selects every depth.
            The historical name is retained for wrapper compatibility.
        cls_token_regularizer_kwargs: Retained transformer-compatible metadata.
        extra_depth_specs: Shape-preserving residual depths previously added by
            :meth:`add_depths`; normally left empty at construction.
        name_prefix: Prefix for generated layer names.
        build: Build all variables immediately for EMA cloning.
        **kwargs: Standard Keras model options.
    """

    FC = "0_feature_connector"
    CB = "1_convolution_block"
    DS = "2_downsampler"
    US = "3_upsampler"
    R = "4_reshaper"
    CTR = "5_cls_token_regularizer"

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
        dropout_rate: float = 0.0, 
        downsampling_method: str = "avg_pooling", 
        upsampling_method: str = "interpolate", 
        upsampling_interpolation: str = "bilinear", 
        use_skip_connections: bool | None = None, 
        reshaper_ids_dict: Mapping[int, str] = {}, 
        reshaper_kwargs: Mapping[str, object] = {}, 
        cls_token_regularizer_ids: Sequence[int | None] = (), 
        cls_token_regularizer_kwargs: Mapping[str, int] = {
            "start": 0, 
            "end": 1, 
        }, 
        extra_depth_specs: Sequence[object] = (), 
        name_prefix: str = "", 
        build: bool = True, 
        **kwargs
    ) -> None:
        widths = tuple(widths)
        reshaper_ids_dict = {
            int(key) if isinstance(key, str) and key.lstrip("-").isdigit()
            else key: value
            for key, value in dict(reshaper_ids_dict).items()
        }
        reshaper_kwargs = dict(reshaper_kwargs)
        cls_token_regularizer_ids = list(cls_token_regularizer_ids)
        cls_token_regularizer_kwargs = dict(cls_token_regularizer_kwargs)
        extra_depth_specs = list(extra_depth_specs)

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

        self.num_labels = self.num_classes + int(self.use_cfg)
        self.condition_dim = self.time_embedding_dim + self.label_embedding_dim
        self.use_reshaper = bool(self.reshaper_ids_dict) or bool(
            self.reshaper_kwargs.get("add_kl", False)
        )
        if self.use_skip_connections is None:
            self.use_skip_connections = not self.use_reshaper
        if self.use_reshaper and self.use_skip_connections:
            raise ValueError(
                "use_skip_connections must be False for a resumable VAE "
                "bottleneck."
            )

        self._base_depth = self._compute_base_depth()
        flatten_depth = 2 * len(self.widths) + 2
        expected_reshapers = {
            flatten_depth: "flatten", 
            flatten_depth + 1: "unflatten", 
        } if self.use_reshaper else {}
        if self.reshaper_ids_dict and self.reshaper_ids_dict != expected_reshapers:
            raise ValueError(
                "reshaper_ids_dict must be the bottleneck pair "
                f"{expected_reshapers}."
            )
        self.reshaper_ids_dict = expected_reshapers
        self.depth = self._base_depth + len(self.extra_depth_specs)
        self.cls_token_regularizer_ids = self._handle_ids(
            self.cls_token_regularizer_ids, 
            depth=self.depth, 
            min_id=0, 
            max_id=self.depth, 
        )
        self.connection_ids_dict: dict[int, list[int]] = {}
        self.cross_attention_ids_dict: dict[int, list[int]] = {}
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
        dropout_rate: float, 
        use_skip_connections: bool | None, 
        reshaper_kwargs: dict, 
        cls_token_regularizer_kwargs: dict, 
    ) -> None:
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

        if not widths or any(
            not isinstance(width, int) or isinstance(width, bool) or width < 1
            for width in widths
        ):
            raise ValueError("widths must contain positive integers.")
        if not isinstance(dropout_rate, (int, float)) or isinstance(
            dropout_rate, bool
        ) or not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in the range [0, 1).")
        if use_skip_connections is not None and not isinstance(
            use_skip_connections, bool
        ):
            raise ValueError("use_skip_connections must be bool or None.")
        unknown_reshaper_keys = set(reshaper_kwargs) - {
            "add_kl", 
            "latent_dim_ratio", 
        }
        if unknown_reshaper_keys:
            raise ValueError(
                f"Unknown reshaper kwargs: {sorted(unknown_reshaper_keys)}."
            )
        add_kl = reshaper_kwargs.get("add_kl", False)
        ratio = reshaper_kwargs.get("latent_dim_ratio", 1.0)
        if not isinstance(add_kl, bool):
            raise ValueError("reshaper add_kl must be boolean.")
        if not isinstance(ratio, (int, float)) or isinstance(
            ratio, bool
        ) or ratio <= 0:
            raise ValueError("latent_dim_ratio must be positive.")
        if set(cls_token_regularizer_kwargs) != {"start", "end"}:
            raise ValueError(
                "cls_token_regularizer_kwargs must contain start and end."
            )

    def _compute_base_depth(self) -> int:
        encoder_depth = 2 * len(self.widths)
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
        max_id: int | None = None, 
    ):
        """Normalize ``None`` and negative IDs like DiffusionTransformer."""

        is_dict = isinstance(ids_dict, dict)
        values_dict = ids_dict if is_dict else {1: list(ids_dict)}
        values_dict = {key: list(value) for key, value in values_dict.items()}
        for key, values in values_dict.items():
            upper = key if max_id is None else max_id
            if None in values:
                values = list(range(min_id, upper + 1))
            fixed = []
            for value in values:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError("Layer IDs must be integers or None.")
                if value < 0:
                    if depth is None:
                        raise ValueError("Negative IDs require a known depth.")
                    value += depth + 1
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
        regularizer_layers = []
        if spatial:
            regularizer_layers.append(
                layers.GlobalAveragePooling2D(name=f"{name}/pool")
            )

        return models.Sequential(
            [
                *regularizer_layers,
                layers.Dense(
                    self.num_classes,
                    activation="softmax",
                    dtype="float32",
                    name=f"{name}/classes",
                ),
            ],
            name=name,
        )

    def _append_stage(
        self, 
        layers_dict: dict[str, layers.Layer], 
        kind: str, 
    ) -> int:
        key = len(self.layers_dicts) + 1
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

    def _residual_stack(self, filters: int, depth: int, name: str):
        return ResidualConvStack(
            filters=filters, 
            depth=depth, 
            condition_dim=self.condition_dim, 
            activation_func=self.activation_func, 
            use_batch_norm=self.use_batch_norm, 
            dropout_rate=self.dropout_rate, 
            name=name, 
            dtype=self.dtype_policy, 
        )

    def _bottleneck_resolution(self, resolution: int) -> int:
        for _ in self.widths:
            resolution = (resolution + 1) // 2
        return resolution

    def _create_layers(self) -> None:
        skip_depths = []
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

        if self.use_reshaper:
            base_side = self._bottleneck_resolution(self.image_size)
            source_shape = (base_side, base_side, self.bottleneck_width)
            flatten_key = len(self.layers_dicts) + 1
            flatten_name = f"{self.name_prefix}depth_{flatten_key}_{self.R[2:]}"
            self._append_stage(
                {
                    self.R: VariationalReshaper(
                        "flatten",  
                        source_shape, 
                        add_kl=bool(self.reshaper_kwargs.get("add_kl", False)), 
                        latent_dim_ratio=float(
                            self.reshaper_kwargs.get("latent_dim_ratio", 1.0)
                        ), 
                        name=flatten_name, 
                        dtype=self.dtype_policy, 
                    )
                }, 
                "flatten", 
            )
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

        if len(self.layers_dicts) != self.depth:
            raise RuntimeError("Constructed U-Net depth does not match its config.")

    def _normalize_extra_spec(self, spec: object) -> dict[str, object]:
        if isinstance(spec, str):
            spec = {spec: True}
        elif isinstance(spec, (tuple, set, frozenset)):
            spec = dict.fromkeys(spec, True)
        if not isinstance(spec, Mapping):
            raise ValueError("A depth specification must be a string or mapping.")
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
            if name not in aliases:
                raise ValueError(f"Unknown progressive U-Net layer: {name}.")
            if options is not False:
                normalized[aliases[name]] = options

        if not normalized:
            raise ValueError("A progressive depth must contain a layer.")

        return normalized

    def _append_extra_stage(self, spec: object) -> int:
        spec = self._normalize_extra_spec(spec)
        key = len(self.layers_dicts) + 1
        stage_layers = {}

        if self.CB in spec:
            stage_layers[self.CB] = self._residual_stack(
                self.widths[0], 
                self.block_depth, 
                f"{self.name_prefix}depth_{key}_{self.CB[2:]}", 
            )

        if self.CTR in spec:
            stage_layers[self.CTR] = self._create_regularizer(
                f"{self.name_prefix}depth_{key}_{self.CTR[2:]}", 
                spatial=True,
            )

        return self._append_stage(stage_layers, "extra")

    def _broadcast_condition(
        self, 
        condition: tf.Tensor, 
        images: tf.Tensor, 
    ) -> tf.Tensor:
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
        training: bool | None = None, 
    ):
        times = tf.convert_to_tensor(times)
        labels = tf.convert_to_tensor(labels)
        time_embeddings = self.time_embedder(times, training=training)
        label_embeddings = self.label_embedder(labels, training=training)
        condition = tf.concat([time_embeddings, label_embeddings], axis=-1)
        condition = tf.cast(condition, self.compute_dtype)

        if full_return:
            return condition, time_embeddings, label_embeddings
        return condition

    def encode(
        self, 
        inputs: UNetInputs, 
        max_depth: int = -1, 
        training: bool | None = None, 
        min_depth: int = 0, 
    ) -> tuple[
        tf.Tensor, 
        tf.Tensor, 
        list[tf.Tensor | None], 
        list[tf.Tensor | None], 
        tuple[tf.Tensor | None, tf.Tensor | None], 
    ]:
        """Run a contiguous range of convolutional depths."""

        if not 0 <= min_depth <= self.depth:
            raise ValueError("min_depth must be in [0, depth].")

        condition, _, label_embeddings = self.embed_conditions(
            inputs[1], 
            inputs[2], 
            full_return=True, 
            training=training, 
        )
        if min_depth == 0:
            x = self.image_embedder(inputs[0], training=training)
            x = tf.concat([x, self._broadcast_condition(condition, x)], axis=-1)
        else:
            x = inputs[0]

        label_reg = self.labels_embed_reg(
            label_embeddings,
            training=training,
        ) if self.labels_embed_reg is not None else None

        features_list: list[tf.Tensor | None] = [None] * min_depth + [x]
        regs_list: list[tf.Tensor | None] = [label_reg] + [None] * min_depth
        z_vals: tuple[tf.Tensor | None, tf.Tensor | None] = (None, None)

        for index, stage in enumerate(self.layers_dicts):
            if index == max_depth:
                break
            if index < min_depth:
                continue

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

            if self.US in stage:
                x = stage[self.US]((x, condition), training=training)

            if self.CB in stage:
                x = stage[self.CB]((x, condition), training=training)

            if self.DS in stage:
                x = stage[self.DS]((x, condition), training=training)

            if self.R in stage:
                reshape_type = self.reshaper_ids_dict[index + 1]
                if reshape_type == "flatten":
                    base_side = self._bottleneck_resolution(self.image_size)
                    x = tf.image.resize(x, (base_side, base_side))

                x, x_mean, x_log_var = stage[self.R](x, training=training)

                if reshape_type == "unflatten":
                    side = self._bottleneck_resolution(self.current_resolution)
                    x = tf.image.resize(x, (side, side))

                if reshape_type == "flatten" and bool(
                    self.reshaper_kwargs.get("add_kl", False)
                ):
                    z_vals = (x_mean, x_log_var)

            regularizer = stage[self.CTR](
                x, 
                training=training,
            ) if self.CTR in stage else None

            features_list.append(x)
            regs_list.append(regularizer)

        return x, condition, features_list, regs_list, z_vals

    def call(
        self, 
        inputs: UNetInputs, 
        full_return: bool = False, 
        training: bool | None = None, 
        min_depth: int = 0, 
    ) -> tf.Tensor | UNetFullOutput:
        x, condition, features_list, regs_list, z_vals = self.encode(
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

        if full_return:
            return predicted_noise, condition, features_list, regs_list, z_vals
        return predicted_noise

    def predict_noise(
        self, 
        inputs: UNetInputs, 
        full_return: bool = False, 
        training: bool | None = None, 
    ):
        return UNet.call(
            self, 
            inputs, 
            full_return=full_return, 
            training=training, 
        )

    @property
    def current_resolution(self) -> int:
        return self._current_resolution

    def set_current_resolution(self, resolution: int | None = None) -> None:
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

    def build(
        self, 
        input_shape: tuple[tf.TensorShape, tf.TensorShape, tf.TensorShape]
        | None = None, 
    ) -> None:
        del input_shape
        shapes = self.build_model()
        super().build(shapes)

    def build_model(self, call_model: bool = True) -> list[tf.TensorShape]:
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

    def add_depths(self, depth_spec: object) -> dict[str, dict[str, int]]:
        """Append shape-preserving convolution or regularizer stages."""

        specs = depth_spec if isinstance(depth_spec, list) else [depth_spec]
        specs = [spec for spec in specs if spec is not None]
        before = self.depth
        if not specs:
            return {"network": {"before": before, "added": 0, "after": before}}

        normalized = [self._normalize_extra_spec(spec) for spec in specs]
        serializable_specs = []
        for spec in normalized:
            saved = {}
            if self.CB in spec:
                saved["convolution_block"] = spec[self.CB]
            if self.CTR in spec:
                saved["cls_token_regularizer"] = spec[self.CTR]
            serializable_specs.append(saved)
            new_depth = self._append_extra_stage(saved)
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
        vars: list[tf.Variable] | None = None, 
    ) -> list[str]:
        """Return TensorFlow names for all or selected trainable variables."""

        vars = self.trainable_variables if vars is None else vars

        return [variable.name for variable in vars]


if __name__ != "__main__":
    tf.keras.utils.register_keras_serializable(
        package="continual_learning",
    )(UNet)
    # TensorFlow 2.10 writes subclassed models to JSON with the bare class
    # name even when a registered package name exists.
    tf.keras.utils.get_custom_objects()["UNet"] = UNet


def run_self_tests() -> dict[str, str]:
    """Run direct, wrapper, VAE, progressive, and serialization checks."""

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
    images = tf.ones((2, 5, 7, 1))
    times = tf.constant([0, 3], tf.int32)
    labels = tf.constant([0, 1], tf.uint8)
    output, condition, features, regs, z_vals = model(
        (images, times, labels), full_return=True
    )
    assert output.shape == images.shape
    assert condition.shape == (2, 4)
    assert len(features) == len(regs) == model.depth + 1
    assert z_vals == (None, None)
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

    vae = UNet(
        **common, 
        reshaper_kwargs={"add_kl": True, "latent_dim_ratio": 0.5}, 
    )
    square_images = images[:, :5, :5]
    vae_output = vae((square_images, times, labels), full_return=True)
    assert vae_output[0].shape == (2, 5, 5, 1)
    assert vae_output[-1][0].shape.rank == 2
    assert vae_output[-1][1].shape == vae_output[-1][0].shape
    assert not vae.connection_ids_dict and vae.reshaper_ids_dict
    json_clone = tf.keras.models.model_from_json(vae.to_json())
    assert json_clone.reshaper_ids_dict == vae.reshaper_ids_dict
    assert json_clone((square_images, times, labels)).shape == square_images.shape


    from diffusion.models.wrapper.diffusion_model import DiffusionModel


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


if __name__ == "__main__":
    print(run_self_tests())
