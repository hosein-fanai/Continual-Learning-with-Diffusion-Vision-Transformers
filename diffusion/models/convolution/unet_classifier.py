"""Convolutional U-Net with a feature-based classifier branch."""

import tensorflow as tf
from tensorflow.keras import layers, models

from copy import deepcopy

from collections.abc import Mapping

from diffusion.layers.convolution import LayerDict
from diffusion.layers.convolution import ResidualConvStack
from diffusion.layers.convolution import VariationalReshaper
from diffusion.models.convolution import UNetFullOutput, UNetInputs
from diffusion.models.convolution.unet import UNet


@tf.keras.utils.register_keras_serializable(package="continual_learning")
class UNetClassifier(UNet):
    """Attach a small convolutional classifier to a conditional U-Net.

    The noise branch is the inherited :class:`UNet`. The classifier reads one
    or more saved U-Net stage features, aligns their spatial sizes, processes them
    with residual convolution stacks, globally pools the final map, and emits
    float32 class probabilities. Its public attributes and return structures
    match the contracts consumed by ``DiffusionClassifier`` and
    ``DiffusionClassifierV2``.
    """

    STACK = "0_residual_conv_stack"
    PROJECTOR = "1_feature_projector"
    REGULARIZER = "2_cls_token_regularizer"
    RESHAPER = "3_reshaper"

    def __init__(
        self, 
        aggregate_from_noises: bool = False, 
        feature_aggregation_ids_dict: dict[int, tuple[int | None, ...]] = {
            1: (-1,), 
        }, 
        classifier_only_cls_token: bool = False, 
        clf_dim: int | None = None, 
        clf_depth: int = 1, 
        clf_block_depth: int = 1, 
        clf_reshaper_kwargs: dict = {}, 
        clf_cls_token_regularizer_ids: list[int | None] = [], 
        force_global_avg_pooling: bool = True, 
        classifier_mlp_ratio: float | None = None, 
        classifier_mlp_activation_func: str = "tanh", 
        dropout_rate: float = 0.0, 
        build: bool = True, 
        **kwargs: object
    ) -> None:
        """Create the inherited denoiser and its convolutional classifier.

        ``feature_aggregation_ids_dict`` maps classifier depths to inherited
        U-Net feature depths. Depth zero is the embedded image and depths
        ``1..depth`` are U-Net stage outputs. Negative IDs are relative to
        the last U-Net depth. Key 1 supplies the initial classifier feature;
        later keys inject additional main features before that classifier
        depth.

        ``clf_reshaper_kwargs`` accepts ``add_kl`` and ``latent_dim_ratio``.
        When KL is enabled, the globally pooled classifier feature is sampled
        through :class:`VariationalReshaper`, and its rank-two statistics are
        returned for the unchanged classifier wrappers.

        Args:
            aggregate_from_noises (bool): Classify the predicted noise image
                instead of saved U-Net features.
            feature_aggregation_ids_dict (dict[int, tuple[int | None, ...]]):
                Classifier depth-to-U-Net feature routes.
            classifier_only_cls_token (bool): Compatibility flag controlling
                ownership of the tracked classifier token placeholder.
            clf_dim (int | None): Positive classifier width; None uses the last
                U-Net encoder width.
            clf_depth (int): Nonnegative number of classifier residual stages.
            clf_block_depth (int): Positive residual blocks per classifier stage.
            clf_reshaper_kwargs (dict[str, object]): Optional classifier
                ``add_kl`` and positive ``latent_dim_ratio``.
            clf_cls_token_regularizer_ids (list[int | None]): Auxiliary
                classifier-head depths; None expands across all depths.
            force_global_avg_pooling (bool): Globally pool classifier maps when
                true; false uses resolution-independent global max pooling.
            classifier_mlp_ratio (float | None): Positive hidden-width ratio,
                or None to omit the hidden Dense layer.
            classifier_mlp_activation_func (str): Hidden Keras activation name.
            dropout_rate (float): Classifier/U-Net dropout probability in
                ``[0,1)``.
            build (bool): Build model variables immediately when true.
            **kwargs (object): Inherited :class:`UNet` constructor options and
                standard Keras model options.

        Returns:
            None: Denoiser, classifier branch, and optional symbolic graph are
            initialized in place.
        """

        feature_aggregation_ids_dict = deepcopy(feature_aggregation_ids_dict)
        # Normalize numeric string depth keys before base-class serialization.
        if isinstance(feature_aggregation_ids_dict, dict):
            feature_aggregation_ids_dict = {
                int(key) if isinstance(key, str) and key.lstrip("-").isdigit()
                else key: value
                for key, value in feature_aggregation_ids_dict.items()
            }
        clf_reshaper_kwargs = deepcopy(clf_reshaper_kwargs)
        clf_cls_token_regularizer_ids = list(clf_cls_token_regularizer_ids)

        super().__init__(
            build=False, 
            dropout_rate=dropout_rate, 
            **kwargs
        )
        self._check_classifier_arguments(locals())
        self._save_init_args(locals())

        self.feature_aggregation_ids_dict = self._handle_ids(
            self.feature_aggregation_ids_dict, 
            depth=self.depth, 
            min_id=0, 
            max_id=self.depth, 
        )
        self.clf_cls_token_regularizer_ids = self._handle_ids(
            self.clf_cls_token_regularizer_ids, 
            depth=self.clf_depth, 
            min_id=0, 
            max_id=self.clf_depth, 
        )

        default_clf_dim = int(self.widths[-1])
        self.clf_dim = default_clf_dim if self.clf_dim is None else self.clf_dim
        self.set_max_encoder_num()

        # The convolutional classifier has no token. This empty tracked layer
        # keeps the legacy V2 variable selector safe if the compatibility flag
        # is explicitly enabled.
        if self.classifier_only_cls_token and getattr(self, "cls_token", None) is None:
            self.cls_token = LayerDict(
                name=f"{self.name_prefix}clf_depth_0_cls_token",
            )

        self.classifier_feature_extractor = (
            layers.GlobalAveragePooling2D(
                dtype=tf.float32,
                name=f"{self.name_prefix}classifier_feature_extractor",
            )
            if self.force_global_avg_pooling
            else layers.GlobalMaxPooling2D(
                dtype=tf.float32,
                name=f"{self.name_prefix}classifier_feature_extractor",
            )
        )
        self.clf_labels_embed_reg = self._create_clf_regularizer(
            "clf_depth_0_cls_token_regularizer",
        ) if 0 in self.clf_cls_token_regularizer_ids else None
        self._create_classifier_layers()
        self._create_classifier_head()

        # Materialize classifier and denoiser variables when requested.
        if self.build_:
            self.build()

    def _check_classifier_arguments(self, local_vars: dict) -> None:
        """Validate dimensions, feature IDs, and classifier-only options.

        Args:
            local_vars (dict[str, object]): Classifier constructor namespace.

        Returns:
            None: Valid inputs return normally; invalid inputs raise ValueError.
        """

        # Require an explicit boolean for noise-based classification.
        if not isinstance(local_vars["aggregate_from_noises"], bool):
            raise ValueError("aggregate_from_noises must be boolean.")
        # Require an explicit boolean for classifier-only token ownership.
        if not isinstance(local_vars["classifier_only_cls_token"], bool):
            raise ValueError("classifier_only_cls_token must be boolean.")
        # Require an explicit boolean for the classifier pooling policy.
        if not isinstance(local_vars["force_global_avg_pooling"], bool):
            raise ValueError("force_global_avg_pooling must be boolean.")

        for name in ("clf_depth", "clf_block_depth"):
            value = local_vars[name]
            minimum = 0 if name == "clf_depth" else 1
            # Validate classifier counts and depth as bounded integers.
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")

        clf_dim = local_vars["clf_dim"]
        # Require an optional classifier width to be a positive integer.
        if clf_dim is not None and (
            not isinstance(clf_dim, int) or isinstance(clf_dim, bool) or clf_dim < 1
        ):
            raise ValueError("clf_dim must be None or a positive integer.")

        mlp_ratio = local_vars["classifier_mlp_ratio"]
        # Require an optional classifier MLP ratio to be positive and numeric.
        if mlp_ratio is not None and (
            not isinstance(mlp_ratio, (int, float))
            or isinstance(mlp_ratio, bool)
            or mlp_ratio <= 0.0
        ):
            raise ValueError("classifier_mlp_ratio must be None or positive.")
        # Keep classifier dropout in the half-open unit interval.
        if not 0.0 <= local_vars["dropout_rate"] < 1.0:
            raise ValueError("dropout_rate must be in the range [0, 1).")

        aggregation_ids = local_vars["feature_aggregation_ids_dict"]
        # Require an initial feature route for classifier depth one.
        if not isinstance(aggregation_ids, dict) or 1 not in aggregation_ids:
            raise ValueError("feature_aggregation_ids_dict must contain key 1.")
        max_classifier_key = max(1, local_vars["clf_depth"])
        for key, ids in aggregation_ids.items():
            # Restrict aggregation keys to valid classifier depths.
            if not isinstance(key, int) or isinstance(key, bool) \
            or not 1 <= key <= max_classifier_key:
                raise ValueError("Classifier aggregation depths are out of range.")
            # Require each aggregation stage to select at least one feature.
            if not isinstance(ids, (list, tuple)) or len(ids) == 0:
                raise ValueError("Every classifier aggregation needs feature IDs.")
            for feature_id in ids:
                # Preserve None as the sentinel for all compatible features.
                if feature_id is None:
                    continue
                # Restrict explicit feature IDs to the encoder's relative-ID range.
                if not isinstance(feature_id, int) or isinstance(feature_id, bool) \
                or not -(self.depth + 1) <= feature_id <= self.depth:
                    raise ValueError("A classifier feature ID is out of range.")

        allowed_reshaper_keys = {"add_kl", "latent_dim_ratio"}
        # Reject unknown or non-mapping classifier reshaper options.
        if not isinstance(local_vars["clf_reshaper_kwargs"], dict) \
        or not set(local_vars["clf_reshaper_kwargs"]) <= allowed_reshaper_keys:
            raise ValueError(
                "clf_reshaper_kwargs accepts only add_kl and latent_dim_ratio."
            )
        add_kl = local_vars["clf_reshaper_kwargs"].get("add_kl", False)
        latent_dim_ratio = local_vars["clf_reshaper_kwargs"].get(
            "latent_dim_ratio", 1.0,
        )
        # Require a boolean classifier KL switch.
        if not isinstance(add_kl, bool):
            raise ValueError("Classifier reshaper add_kl must be boolean.")
        # Require a positive classifier latent-width ratio.
        if not isinstance(latent_dim_ratio, (int, float)) \
        or isinstance(latent_dim_ratio, bool) or latent_dim_ratio <= 0.0:
            raise ValueError("Classifier latent_dim_ratio must be positive.")

    def _create_clf_regularizer(self, suffix: str) -> layers.Layer:
        """Create a float32 auxiliary classifier head.

        Args:
            suffix (str): Suffix appended to the model name prefix.

        Returns:
            tf.keras.layers.Layer: Dense class-probability projection.
        """

        return layers.Dense(
            self.num_classes, 
            activation="softmax", 
            dtype=tf.float32, 
            name=f"{self.name_prefix}{suffix}", 
        )

    def _create_classifier_layers(self) -> None:
        """Create tracked convolution stages and one terminal layer mapping.

        Returns:
            None: ``clf_layers_dicts`` is populated in place.
        """

        self.clf_layers_dicts = []
        for depth_id in range(1, self.clf_depth + 1):
            self.clf_layers_dicts.append(
                self._make_classifier_stage(depth_id)
            )

        terminal = LayerDict(
            name=f"{self.name_prefix}clf_terminal",
        )
        terminal[self.PROJECTOR] = layers.Conv2D(
            self.clf_dim, 
            kernel_size=1, 
            dtype=tf.float32, 
            name=f"{self.name_prefix}clf_terminal_feature_projector",
        )
        # Add a variational classifier bottleneck when KL output is enabled.
        if self.clf_reshaper_kwargs.get("add_kl", False):
            terminal[self.RESHAPER] = VariationalReshaper(
                reshape_type="flatten", 
                source_shape=(self.clf_dim,), 
                add_kl=True, 
                latent_dim_ratio=self.clf_reshaper_kwargs.get(
                    "latent_dim_ratio", 1.0, 
                ), 
                dtype=tf.float32, 
                name=f"{self.name_prefix}clf_terminal_reshaper", 
            )
        self.clf_layers_dicts.append(terminal)

    def _make_classifier_stage(self, depth_id: int) -> LayerDict:
        """Create one fixed-width classifier processing depth.

        Args:
            depth_id (int): Positive one-based classifier depth.

        Returns:
            LayerDict: Residual stack and optional auxiliary regularizer.
        """

        stage = LayerDict(name=f"{self.name_prefix}clf_depth_{depth_id}")
        stage[self.STACK] = ResidualConvStack(
            filters=self.clf_dim, 
            depth=self.clf_block_depth, 
            condition_dim=getattr(self, "condition_dim", None), 
            activation_func=self.activation_func, 
            use_batch_norm=self.use_batch_norm, 
            name=f"{self.name_prefix}clf_depth_{depth_id}_residual_conv_stack", 
        )
        # Attach an auxiliary regularizer at selected classifier depths.
        if depth_id in self.clf_cls_token_regularizer_ids:
            stage[self.REGULARIZER] = self._create_clf_regularizer(
                f"clf_depth_{depth_id}_cls_token_regularizer", 
            )

        return stage

    def _create_classifier_head(self) -> None:
        """Create the optional hidden projection and final softmax.

        Returns:
            None: ``classifier`` is assigned in place.
        """

        self.classifier = models.Sequential(
            name=f"{self.name_prefix}classes",
        )
        # Add the optional hidden classifier projection.
        if self.classifier_mlp_ratio is not None:
            self.classifier.add(layers.Dense(
                max(1, int(self.clf_dim * self.classifier_mlp_ratio)),
                activation=self.classifier_mlp_activation_func,
                dtype=tf.float32,
                name=f"{self.name_prefix}classes_first_layer",
            ))
        # Add classifier dropout only for a nonzero rate.
        if self.dropout_rate > 0.0:
            self.classifier.add(layers.Dropout(
                self.dropout_rate,
                dtype=tf.float32,
                name=f"{self.name_prefix}classes_dropout",
            ))
        self.classifier.add(layers.Dense(
            self.num_classes,
            activation="softmax",
            dtype=tf.float32,
            name=f"{self.name_prefix}classes_final_layer",
        ))

    def add_class(self, source_network: object | None = None) -> None:
        """Append one classifier output while preserving the existing head.

        Args:
            source_network (object | None): Optional already-expanded raw
                classifier whose new output initializes an EMA clone.

        Returns:
            None: The label vocabulary and final classifier layer grow by one.
        """

        old_layer = self.classifier.layers[-1]
        old_kernel, old_bias = old_layer.get_weights()
        super().add_class(source_network=source_network)

        self.clf_labels_embed_reg = self._expand_regularizer(
            self.clf_labels_embed_reg,
            source_network.clf_labels_embed_reg
            if source_network is not None else None,
        )
        for index, stage in enumerate(self.clf_layers_dicts):
            # Expand only classifier stages that own an auxiliary class head.
            if self.REGULARIZER in stage:
                stage[self.REGULARIZER] = self._expand_regularizer(
                    stage[self.REGULARIZER],
                    source_network.clf_layers_dicts[index][self.REGULARIZER]
                    if source_network is not None else None,
                )

        layer_config = old_layer.get_config()
        layer_config["units"] = self.num_classes
        new_layer = old_layer.__class__.from_config(layer_config)
        new_layer.build((None, old_kernel.shape[0]))
        new_kernel, new_bias = new_layer.get_weights()
        new_kernel[..., :-1] = old_kernel
        new_bias[:-1] = old_bias

        # Initialize the new EMA output from the already-expanded raw network.
        if source_network is not None:
            source_kernel, source_bias = (
                source_network.classifier.layers[-1].get_weights()
            )
            new_kernel[..., -1] = source_kernel[..., -1]
            new_bias[-1] = source_bias[-1]

        new_layer.set_weights([new_kernel, new_bias])
        self.classifier.pop()
        self.classifier.add(new_layer)

    def set_max_encoder_num(self, max_encoder_num: int | None = None) -> None:
        """Set the greatest inherited feature depth needed for classification.

        Args:
            max_encoder_num (int | None): Depth in ``[0, depth]``; None infers
                the greatest routed main feature.

        Returns:
            None: ``max_encoder_num`` is updated in place.
        """

        # Infer the deepest encoder feature required by configured routes.
        if max_encoder_num is None:
            feature_ids = [
                feature_id
                for ids in self.feature_aggregation_ids_dict.values()
                for feature_id in ids
            ]
            max_encoder_num = self.depth if self.aggregate_from_noises \
                else max(feature_ids)
        # Require an explicit encoder limit to be a valid network depth.
        if not isinstance(max_encoder_num, int) or isinstance(max_encoder_num, bool) \
        or not 0 <= max_encoder_num <= self.depth:
            raise ValueError("max_encoder_num must be in the range [0, depth].")
        self.max_encoder_num = max_encoder_num

    @staticmethod
    def _resize_feature(feature: tf.Tensor, reference: tf.Tensor) -> tf.Tensor:
        """Resize one feature map to a reference feature's spatial shape.

        Args:
            feature (tf.Tensor): Rank-four source feature map.
            reference (tf.Tensor): Rank-four target-shape feature map.

        Returns:
            tf.Tensor: Resized source in the reference dtype.
        """

        # Align only spatial feature maps with known image axes.
        if feature.shape.rank != 4 or reference.shape.rank != 4:
            raise ValueError("Classifier features must be rank-four image maps.")

        feature = tf.image.resize(
            feature, 
            tf.shape(reference)[1: 3]
        )

        return tf.cast(feature, reference.dtype)

    def _normalize_feature(self, feature: tf.Tensor) -> tf.Tensor:
        """Return a rank-four feature with the fixed classifier width.

        Args:
            feature (tf.Tensor): Rank-two vector or rank-four image feature.

        Returns:
            tf.Tensor: Rank-four feature padded/truncated to ``clf_dim``.
        """

        # Promote vector features to one-by-one spatial maps for aggregation.
        if feature.shape.rank == 2:
            feature = feature[:, None, None, :]
        # Reject feature ranks unsupported by classifier aggregation.
        elif feature.shape.rank != 4:
            raise ValueError("Classifier features must be rank two or four.")

        feature = feature[..., :self.clf_dim]
        channel_padding = tf.maximum(
            self.clf_dim - tf.shape(feature)[-1], 
            0, 
        )
        paddings = tf.concat([
            tf.zeros((3, 2), dtype=tf.int32), 
            tf.reshape(tf.stack([0, channel_padding]), (1, 2)), 
        ], axis=0)
        feature = tf.pad(feature, paddings)
        feature.set_shape(feature.shape[:-1].concatenate([self.clf_dim]))

        return feature

    def _aggregate_features(
        self, 
        features_list: list[tf.Tensor], 
        feature_ids: list[int], 
        current: tf.Tensor | None = None
    ) -> tf.Tensor:
        """Align and average selected main features and current state.

        Args:
            features_list (list[tf.Tensor]): Main-branch depth features.
            feature_ids (list[int]): Selected absolute feature depths.
            current (tf.Tensor | None): Current classifier feature to include.

        Returns:
            tf.Tensor: Spatially aligned, fixed-width average feature.
        """

        selected = [features_list[feature_id] for feature_id in feature_ids]
        # Fail if an explicitly requested encoder feature was not produced.
        if any(feature is None for feature in selected):
            raise ValueError("A requested classifier feature was not encoded.")
        selected = [self._normalize_feature(feature) for feature in selected]

        # Initialize aggregation from the deepest selected feature group.
        if current is None:
            reference = selected[-1]
            aligned = [
                feature if feature is reference else self._resize_feature(
                    feature, reference,
                )
                for feature in selected
            ]

            return aligned[0] if len(aligned) == 1 else tf.add_n(aligned) / len(
                aligned
            )

        current = self._normalize_feature(current)
        aligned = [self._resize_feature(feature, current) for feature in selected]
        aligned = [current, *aligned]

        return tf.add_n(aligned) / len(aligned)

    def _regularize_feature(
        self, 
        regularizer: layers.Layer | None, 
        feature: tf.Tensor, 
        training: bool | None
    ) -> tf.Tensor | None:
        """Pool a feature and apply an optional auxiliary classifier.

        Args:
            regularizer (tf.keras.layers.Layer | None): Optional class head.
            feature (tf.Tensor): Rank-four classifier feature.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor | None: Float32 class probabilities or None.
        """

        # Skip auxiliary prediction when this depth has no regularizer.
        if regularizer is None:
            return None

        pooled = self.classifier_feature_extractor(
            feature, 
            training=training
        )

        return tf.cast(
            regularizer(pooled, training=training), 
            tf.float32
        )

    def compute_class(
        self, 
        features_list: list[tf.Tensor], 
        noises: tf.Tensor | None, 
        times: tf.Tensor, 
        labels: tf.Tensor, 
        cond: tf.Tensor | None = None, 
        training: bool | None = None
    ) -> tuple[
        tf.Tensor, 
        tf.Tensor | None, 
        list[tf.Tensor], 
        list[tf.Tensor | None], 
        tuple[tf.Tensor | None, tf.Tensor | None],
    ]:
        """Return class probabilities and classifier branch intermediates.

        Args:
            features_list (list[tf.Tensor]): Saved main U-Net features.
            noises (tf.Tensor | None): Predicted noise image when classifying it.
            times (tf.Tensor): Integer timestep IDs; retained for API parity.
            labels (tf.Tensor): Integer label IDs; retained for API parity.
            cond (tf.Tensor | None): Condition passed into residual stacks.
            training (bool | None): Keras training mode.

        Returns:
            tuple[tf.Tensor, tf.Tensor | None, list[tf.Tensor],
            list[tf.Tensor | None], tuple[tf.Tensor | None, tf.Tensor | None]]:
            Probabilities, condition, classifier features, auxiliary predictions,
            and classifier latent mean/log variance.
        """

        del times, labels
        # Start classification from predicted noise when configured.
        if self.aggregate_from_noises:
            # Noise-based aggregation requires the denoiser's prediction.
            if noises is None:
                raise ValueError("aggregate_from_noises requires predicted noises.")
            x = self._normalize_feature(noises)
        # Otherwise initialize from configured encoder features.
        else:
            x = self._aggregate_features(
                features_list,
                self.feature_aggregation_ids_dict[1],
            )

        clf_features_list = [x]
        clf_regs_list = [self._regularize_feature(
            self.clf_labels_embed_reg,
            x, 
            training, 
        )]

        for depth_id, stage in enumerate(self.clf_layers_dicts[:-1], start=1):
            # Merge additional encoder features at routed classifier depths.
            if depth_id > 1 and depth_id in self.feature_aggregation_ids_dict:
                x = self._aggregate_features(
                    features_list,
                    self.feature_aggregation_ids_dict[depth_id],
                    current=x,
                )
            x = stage[self.STACK](
                (x, cond) if cond is not None else x,
                training=training,
            )

            clf_features_list.append(x)
            clf_regs_list.append(self._regularize_feature(
                stage.get(self.REGULARIZER), 
                x, 
                training, 
            ))

        terminal = self.clf_layers_dicts[-1]
        # Project terminal features to the configured classifier width.
        if self.PROJECTOR in terminal:
            x = terminal[self.PROJECTOR](x, training=training)

        x = self.classifier_feature_extractor(
            x, 
            training=training
        )
        clf_z_vals = (None, None)

        # Produce classifier latent statistics at a variational terminal.
        if self.RESHAPER in terminal:
            x, z_mean, z_log_var = terminal[self.RESHAPER](x, training=training)
            clf_z_vals = (z_mean, z_log_var)

        clf_features_list.append(x)
        clf_regs_list.append(None)

        classes = tf.cast(
            self.classifier(
                x, 
                training=training
            ), 
            tf.float32
        )

        return classes, cond, clf_features_list, clf_regs_list, clf_z_vals

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        full_return: bool = False, 
        training: bool | None = None, 
        min_depth: int = 0
    ) -> dict[str, object] | tf.Tensor | tuple:
        """Predict both branches, or resume only latent noise decoding.

        ``min_depth=0`` returns the classifier wrapper mapping. A positive
        ``min_depth`` follows :class:`UNet` and returns its tensor/tuple output,
        which lets the unchanged VAE sampler resume after a flatten depth.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Image/latent,
                timestep, and label tensors.
            full_return (bool): Include both branches' intermediates when true.
            training (bool | None): Keras training mode.
            min_depth (int): Resume the denoiser from this depth; zero runs both
                denoiser and classifier.

        Returns:
            dict[str, object] | tf.Tensor | tuple: Branch mapping at depth zero,
            or inherited U-Net tensor/full output when resuming a latent decode.
        """

        # Delegate resumed denoiser execution directly to the base U-Net.
        if min_depth != 0:
            return UNet.call(
                self, 
                inputs, 
                full_return=full_return, 
                training=training, 
                min_depth=min_depth, 
            )

        noises, cond, features_list, regs_list, z_vals = super().call(
            inputs, 
            full_return=True, 
            training=training, 
        )
        classes, clf_cond, clf_features, clf_regs, clf_z_vals = self.compute_class(
            features_list, 
            noises, 
            times=inputs[1], 
            labels=inputs[2], 
            cond=cond, 
            training=training, 
        )
        outputs = {
            "noises": noises, 
            "classes": classes, 
        }

        # Attach condition and intermediate metadata only for full returns.
        if full_return:
            outputs.update({
                "cond": cond, 
                "features_list": features_list, 
                "regs_list": regs_list, 
                "z_vals": z_vals, 
                "clf_cond": clf_cond, 
                "clf_features_list": clf_features, 
                "clf_regs_list": clf_regs, 
                "clf_z_vals": clf_z_vals, 
            })
        return outputs

    def predict_noise(
        self, 
        inputs: UNetInputs,
        full_return: bool = False, 
        training: bool | None = None
    ) -> tf.Tensor | UNetFullOutput:
        """Run only the inherited U-Net noise branch.

        Args:
            inputs (UNetInputs): Image, timestep, and label tensors.
            full_return (bool): Include denoiser intermediates when true.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor | UNetFullOutput: Same output as :meth:`UNet.call`.
        """

        return super().call(
            inputs, 
            full_return=full_return, 
            training=training, 
        )

    def predict_class(
        self, 
        inputs: UNetInputs,
        max_encoder_num: int | None = -1, 
        full_return: bool = False, 
        training: bool | None = None
    ) -> tf.Tensor | tuple[
        tf.Tensor, tf.Tensor | None, list[tf.Tensor],
        list[tf.Tensor | None], tuple[tf.Tensor | None, tf.Tensor | None]
    ]:
        """Classify inputs while executing selected encoder depths.

        Args:
            inputs (UNetInputs): Image, timestep, and label tensors.
            max_encoder_num (int | None): Exclusive encoder stop; -1 runs all
                stages and None uses the greatest routed feature depth.
            full_return (bool): Include classifier intermediates when true.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor | tuple: Float32 probabilities ``[B,num_classes]`` or
            probabilities plus classifier condition, features, regularizers,
            and latent statistics.
        """

        # Run the complete denoiser first when classification consumes its noise output.
        if self.aggregate_from_noises:
            noises, cond, features_list, _, _ = super().call(
                inputs, 
                full_return=True, 
                training=training, 
            )
        # Otherwise encode only as deeply as classifier feature routes require.
        else:
            max_encoder_num = self.max_encoder_num \
                if max_encoder_num is None else max_encoder_num
            _, cond, features_list, _, _ = self.encode(
                inputs, 
                max_depth=max_encoder_num, 
                training=training, 
            )
            noises = None

        outputs = self.compute_class(
            features_list, 
            noises, 
            times=inputs[1], 
            labels=inputs[2], 
            cond=cond, 
            training=training, 
        )

        return outputs if full_return else outputs[0]

    @staticmethod
    def _normalize_classifier_depth_spec(spec: object) -> bool:
        """Validate one progressive classifier depth and report regularization.

        Args:
            spec (object): Layer name, collection, or enabled-option mapping.

        Returns:
            bool: Whether the stage requests an auxiliary regularizer.
        """

        stack_names = {
            "convolution_block", 
            "residual_block", 
            "residual_conv_stack", 
            "vision_transformer_block", 
            UNetClassifier.STACK, 
        }
        regularizer_names = {
            "cls_token_regularizer", 
            UNetClassifier.REGULARIZER, 
        }
        allowed_names = stack_names | regularizer_names

        # Interpret a string as one classifier layer name.
        if isinstance(spec, str):
            names = {spec}
        # Interpret a collection as several layers in one classifier depth.
        elif isinstance(spec, (tuple, set, frozenset)):
            names = set(spec)
        # Retain enabled layer names from a mapped depth specification.
        elif isinstance(spec, Mapping):
            names = {name for name, enabled in spec.items() if enabled is not False}
        # Reject unsupported classifier depth specification types.
        else:
            raise ValueError(
                "A classifier depth must be a layer name, collection, or mapping."
            )

        # Reject empty specifications and unknown classifier layer names.
        if not names or not names <= allowed_names:
            unknown = sorted(str(name) for name in names - allowed_names)
            raise ValueError(f"Unknown progressive classifier layers: {unknown}.")

        return bool(names & regularizer_names)

    def _append_classifier_depth(self, use_regularizer: bool) -> int:
        """Insert one classifier stage immediately before its terminal mapping.

        Args:
            use_regularizer (bool): Attach an auxiliary class head when true.

        Returns:
            int: One-based depth assigned to the new classifier stage.
        """

        terminal = self.clf_layers_dicts.pop()
        new_depth = self.clf_depth + 1
        # Register newly added classifier regularizer depths.
        if use_regularizer:
            self.clf_cls_token_regularizer_ids.append(new_depth)

        self.clf_depth = new_depth
        self.clf_layers_dicts.append(self._make_classifier_stage(new_depth))
        self.clf_layers_dicts.append(terminal)

        return new_depth

    def _refresh_aggregation_ids(self) -> None:
        """Re-resolve negative main-feature IDs after inherited depth growth.

        Returns:
            None: Feature routes and required encoder depth are updated.
        """

        original_ids = deepcopy(
            self._init_config["feature_aggregation_ids_dict"],
        )
        self.feature_aggregation_ids_dict = self._handle_ids(
            original_ids,
            depth=self.depth,
            min_id=0,
            max_id=self.depth,
        )
        self.set_max_encoder_num()

    def add_depths(self, depth_spec: object) -> dict[str, dict[str, int]]:
        """Grow the inherited network, classifier branch, or both.

        Ordinary specifications are delegated to :class:`UNet`. A targeted
        mapping may contain ``network`` and/or ``classifier``. Classifier list
        items append separate fixed-width residual stages; a tuple, set, or
        mapping describes one stage. Transformer block names are accepted as
        compatibility aliases for a residual convolution stack.

        Args:
            depth_spec (object): Untargeted network spec or mapping containing
                ``network`` and/or ``classifier`` specifications.

        Returns:
            dict[str, dict[str, int]]: Per-branch before/added/after counts.
        """

        targeted = isinstance(depth_spec, Mapping) and any(
            name in depth_spec for name in ("network", "classifier")
        )
        # Treat an unscoped specification as ordinary denoiser growth.
        if not targeted:
            growth = super().add_depths(depth_spec)
            self._refresh_aggregation_ids()

            return growth

        # Restrict targeted growth to network and classifier branches.
        if not set(depth_spec) <= {"network", "classifier"}:
            raise ValueError("Targeted depth keys must be network or classifier.")

        classifier_spec = depth_spec.get("classifier")
        # Normalize an empty classifier request to no classifier growth.
        if classifier_spec is None or classifier_spec == [] or classifier_spec == {}:
            classifier_specs = []
        # Preserve a list as an ordered sequence of classifier depths.
        elif isinstance(classifier_spec, list):
            classifier_specs = classifier_spec
        # Wrap one classifier specification as a single-depth sequence.
        else:
            classifier_specs = [classifier_spec]

        before = self.clf_depth
        normalized_specs = [
            self._normalize_classifier_depth_spec(spec)
            for spec in classifier_specs
        ]

        # Validate both branches before changing either topology. The base
        # parser performs its own complete validation before appending stages.
        network_spec = depth_spec.get("network")
        network_growth = super().add_depths(network_spec)
        self._refresh_aggregation_ids()

        for use_regularizer in normalized_specs:
            self._append_classifier_depth(use_regularizer)

        self._init_config["clf_depth"] = self.clf_depth
        self._init_config["clf_cls_token_regularizer_ids"] = list(
            self.clf_cls_token_regularizer_ids,
        )
        self.set_max_encoder_num()
        self.train_function = None
        self.test_function = None
        self.predict_function = None
        return {
            "network": network_growth["network"],
            "classifier": {
                "before": before,
                "added": self.clf_depth - before,
                "after": self.clf_depth,
            },
        }


# TensorFlow 2.10 writes the plain root name for subclassed model JSON.
tf.keras.utils.get_custom_objects()["UNetClassifier"] = UNetClassifier


def run_self_tests() -> dict[str, str]:
    """Run compact call, KL, gradient, and serialization checks.

    Returns:
        dict[str, str]: ``{"UNetClassifier": "passed"}`` after all checks.
    """

    tf.keras.backend.clear_session()
    tf.random.set_seed(109)
    common = {
        "num_classes": 2, 
        "use_cfg": True, 
        "timesteps": 4, 
        "image_size": 4, 
        "channels": 1, 
        "widths": (2,), 
        "block_depth": 1, 
        "bottleneck_width": 3, 
        "bottleneck_depth": 1, 
        "image_embedding_dim": 2, 
        "time_embedding_dim": 3, 
        "label_embedding_dim": 2, 
    }
    inputs = (
        tf.ones((2, 4, 4, 1)), 
        tf.constant([0, 3]), 
        tf.constant([0, 2]), 
    )

    model = UNetClassifier(
        **common, 
        feature_aggregation_ids_dict={1: (-2,)}, 
    )
    outputs = model(inputs, full_return=True, training=False)
    assert set(outputs) == {
        "noises", "cond", "features_list", "regs_list", "z_vals", 
        "classes", "clf_cond", "clf_features_list", "clf_regs_list", 
        "clf_z_vals", 
    }
    assert outputs["noises"].shape == inputs[0].shape
    assert outputs["classes"].shape == (2, 2)
    assert outputs["classes"].dtype == tf.float32
    tf.debugging.assert_near(
        tf.reduce_sum(outputs["classes"], axis=-1), 
        tf.ones((2,)), 
    )
    assert model.predict_noise(inputs).shape == inputs[0].shape
    assert model.predict_class(inputs).shape == (2, 2)
    assert len(model.clf_layers_dicts) == model.clf_depth + 1

    dynamic_regularized = UNetClassifier(
        **{**common, "num_classes": None},
        cls_token_regularizer_ids=[None],
        clf_cls_token_regularizer_ids=[None],
    )
    dynamic_regularized.add_class()
    dynamic_regularized.add_class()
    dynamic_outputs = dynamic_regularized(
        inputs,
        full_return=True,
        training=False,
    )
    assert dynamic_outputs["classes"].shape == (2, 2)
    assert all(item.shape == (2, 2) for item in dynamic_outputs["regs_list"])
    assert all(
        item.shape == (2, 2)
        for item in dynamic_outputs["clf_regs_list"][:-1]
    )

    max_pooled = UNetClassifier(
        **common,
        force_global_avg_pooling=False,
    )
    assert isinstance(
        max_pooled.classifier_feature_extractor,
        layers.GlobalMaxPooling2D,
    )
    assert max_pooled.predict_class(inputs).shape == (2, 2)

    with tf.GradientTape() as tape:
        classes = model.predict_class(inputs, training=True)
        loss = tf.reduce_sum(classes[:, 0])
    gradients = tape.gradient(loss, model.classifier.trainable_variables)
    assert gradients and all(gradient is not None for gradient in gradients)

    variational = UNetClassifier(
        **common, 
        clf_reshaper_kwargs={"add_kl": True, "latent_dim_ratio": 0.5},
    )
    variational_output = variational(inputs, full_return=True, training=False)
    assert variational_output["clf_z_vals"][0].shape == (2, 1)
    assert variational_output["clf_z_vals"][1].shape == (2, 1)

    restored = UNetClassifier.from_config(model.get_config())
    assert restored(inputs)["classes"].shape == (2, 2)
    assert len(restored.weights) == len(model.weights)
    json_restored = tf.keras.models.model_from_json(model.to_json())
    assert json_restored(inputs)["classes"].shape == (2, 2)

    growth = model.add_depths({
        "network": "convolution_block", 
        "classifier": "vision_transformer_block", 
    })
    assert growth["network"]["added"] == 1
    assert growth["classifier"] == {"before": 1, "added": 1, "after": 2}
    assert len(model.clf_layers_dicts) == model.clf_depth + 1
    model.build_model()
    grown_clone = UNetClassifier.from_config(model.get_config())
    assert grown_clone(inputs)["classes"].shape == (2, 2)
    assert len(grown_clone.weights) == len(model.weights)
    assert all(
        left.shape == right.shape
        for left, right in zip(model.weights, grown_clone.weights)
    )

    main_variational = UNetClassifier(
        **common, 
        reshaper_kwargs={"add_kl": True, "latent_dim_ratio": 0.5},
    )


    from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier


    wrapper = DiffusionClassifier(
        network=main_variational, 
        use_ema=False, 
        test_network_name="raw", 
        scheduler_name="linear", 
        test_steps=2, 
        mask_by_nulls=False, 
        p_uncond=0.0, 
    )
    samples = wrapper.sample_vae(network_name="raw", labels=[0, 1], seed=7)
    assert samples.shape == inputs[0].shape

    tf.keras.backend.clear_session()
    return {"UNetClassifier": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
