"""Joint encoder-decoder denoiser with the standard DiT classifier API.

The model uses its inherited :class:`DiTClassifier` state as the encoder and
classifier branch, then feeds the encoder condition and depth-indexed features
to an attached :class:`DiTDecoder`.  Standard three-tensor calls remain
compatible with diffusion wrappers; a fourth image enables teacher forcing.
"""

import tensorflow as tf
from tensorflow.keras import layers

from copy import deepcopy

from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.transformer.di_t_decoder import DiTDecoder
from diffusion.models.transformer.di_t_encoder_decoder import DiTEncoderDecoder


class DiTEncoderDecoderClassifier(DiTEncoderDecoder, DiTClassifier):
    """Combine a classifier encoder with a context-aware diffusion decoder.

    The object itself is initialized as a complete :class:`DiTClassifier`.
    Therefore every inherited transformer and classifier API, including
    :meth:`encode`, :meth:`compute_class`, classifier-only prediction,
    progressive :meth:`add_depths`, and variable inspection, operates normally
    on the encoder/classifier state.  :attr:`encoder` is a read-only alias for
    the object itself.  :attr:`decoder` is a separately configured
    :class:`DiTDecoder`.

    :meth:`call` and :meth:`predict_noise` accept either ``(images, times,
    labels)`` or ``(images, times, labels, decoder_images)``.  Three inputs use
    ``images`` as the decoder input, preserving compatibility with
    ``DiffusionClassifier`` and ``DiffusionClassifierV2``.  Four inputs select
    explicit teacher forcing.  Internally the decoder receives its unique
    arguments separately: ``(decoder_images, times, labels)``, the encoder
    condition, and the encoder feature list.  With
    ``decoder_separate_cond=True``, the decoder instead embeds its own times
    and labels.  A conditionless encoder supplies a zero-valued decoder context
    because the functional decoder head still requires a tensor input.  At
    decoder depths 1..N, each decoder block cross-attends to the final encoder
    feature by default; decoder depth 0 has no context-attention block.

    Progressive depth specifications retain :class:`DiTClassifier` semantics:
    an ordinary specification grows the encoder transformer, while a targeted
    mapping may contain ``"network"``, ``"classifier"``, and ``"decoder"``.
    ``model.decoder`` also retains its explicit
    ``(decoder_inputs, encoder_cond, encoder_features_list)`` interface.

    Attributes:
        encoder (DiTClassifier): Read-only alias returning ``self``.
        decoder (DiTDecoder): Decoder receiving encoder condition/features.
        supports_teacher_forcing (bool): Always ``True``.
    """

    def __init__(
        self, 
        encoder_kwargs: dict[str, object] | None = None, 
        decoder_kwargs: dict[str, object] | None = None, 
        build: bool = True, 
        **kwargs: object
    ) -> None:
        """Initialize the encoder/classifier state and attached decoder.

        Args:
            encoder_kwargs (dict[str, object] | None): Optional nested
                :class:`DiTClassifier` arguments.  Every
                ``DiffusionTransformer`` and ``DiTClassifier`` constructor key
                is valid, including all routing dictionaries, component option
                dictionaries, ``clf_*`` settings, dimensions, conditions,
                tokens, and Keras model options.  Flat values in ``**kwargs``
                override equal nested keys.  ``None`` means use flat arguments
                and classifier defaults.
                ``use_unpatchify`` is forced to false because the decoder owns
                the composite output head.  For noise-based classification the
                parent validation is satisfied during initialization without
                retaining or building its otherwise unused output layer.
            decoder_kwargs (dict[str, object] | None): :class:`DiTDecoder`
                arguments. Encoder feature dimensions and grids are derived
                from the actual classifier encoder; explicitly supplied
                metadata must match. Shared image, class, timestep, patch,
                token-width, and condition-width settings default to the
                encoder values.  Decoder symbolic building is managed by this
                composite, so a supplied ``build`` value is ignored.
                ``shift_inputs`` defaults to ``False`` for wrapper-compatible
                denoising; pass ``True`` to right-shift teacher-forcing tokens.
                The outer dtype policy is inherited unless ``dtype`` is set
                explicitly here. Image size/channels must match the encoder and
                ``use_unpatchify`` must be true. Configure KL bottlenecks and
                token regularizers on the encoder, where unchanged classifier
                wrappers read their loss metadata. ``cond_dim`` must match the
                encoder unless ``decoder_separate_cond=True``; any decoder
                timestep/label tables must cover the encoder ID ranges.
                Feature-width merges and encoder features used as cross-
                attention queries require matching encoder/decoder class-token
                presence; attention values may differ in length.
            build (bool): Build a four-input symbolic graph immediately.
                ``False`` defers it until Keras first calls the model or until
                :meth:`build` is invoked explicitly.
            **kwargs (object): Flat ``DiffusionTransformer``/``DiTClassifier``
                arguments plus standard Keras ``Model`` keys such as ``name``,
                ``dtype``, ``trainable``, and ``dynamic``.  Flat keys take
                precedence over ``encoder_kwargs``.

        Returns:
            None: Encoder/classifier layers, the decoder, serialization state,
            and optionally the four-input symbolic graph are initialized.

        Raises:
            ValueError: If decoder encoder-feature metadata contradicts the
                constructed encoder, its output violates the wrapper image
                contract, or decoder-only auxiliary losses are requested.
        """
        saved_encoder_kwargs = (
            {} if encoder_kwargs is None else deepcopy(encoder_kwargs)
        )
        saved_decoder_kwargs = (
            {} if decoder_kwargs is None else deepcopy(decoder_kwargs)
        )

        classifier_kwargs = deepcopy(saved_encoder_kwargs)
        classifier_kwargs.update(deepcopy(kwargs))
        classifier_kwargs["build"] = False
        aggregate_from_noises = bool(
            classifier_kwargs.get("aggregate_from_noises", False)
        )
        classifier_kwargs["use_unpatchify"] = aggregate_from_noises
        classifier_kwargs.setdefault("name_prefix", "encoder_model/")
        classifier_kwargs.setdefault(
            "feature_aggregation_ids_dict", {1: (-1,)}
        )
        classifier_kwargs.setdefault(
            "clf_connection_ids_dict", {-1: (-1,)}
        )
        DiTClassifier.__init__(self, **classifier_kwargs)
        self.unpatchifier = None
        self.use_unpatchify = False
        self._init_config["use_unpatchify"] = False

        decoder_config = deepcopy(saved_decoder_kwargs)
        decoder_config["build"] = False
        decoder_config.setdefault("name_prefix", "decoder_model/")
        # Keep the nested decoder on the same dynamic vocabulary contract.
        if self.dynamic_num_classes:
            decoder_config["num_classes"] = None
            decoder_config["use_cfg"] = self.use_cfg
        # Preserve explicit nested-decoder settings in fixed-width mode.
        else:
            decoder_config.setdefault("num_classes", self.num_classes)
            decoder_config.setdefault("use_cfg", self.use_cfg)
        decoder_config.setdefault("timesteps", self.timesteps)
        decoder_config.setdefault("image_size", self.image_size)
        decoder_config.setdefault("channels", self.channels)
        decoder_config.setdefault("patch_size", self.patch_size)
        decoder_config.setdefault("dim", self.dim)
        decoder_config.setdefault("cond_dim", self.cond_dim)
        decoder_config.setdefault("dtype", self.dtype_policy.name)
        decoder_config.setdefault("shift_inputs", False)
        encoder_feature_dims, encoder_feature_grids = \
            DiTEncoderDecoder._get_encoder_feature_metadata(self)
        # Require a spatial final encoder feature for decoder initialization.
        if encoder_feature_grids[-1] is None:
            raise ValueError(
                "the encoder's final feature must be spatial for DiTDecoder."
            )
        inferred_metadata = {
            "encoder_feature_dims": encoder_feature_dims, 
            "encoder_feature_grid_sizes": encoder_feature_grids, 
            "encoder_output_grid_size": encoder_feature_grids[-1], 
            "encoder_output_dim": encoder_feature_dims[-1], 
        }
        for key, value in inferred_metadata.items():
            supplied = decoder_config.get(key)
            matches = (
                isinstance(supplied, (list, tuple))
                and list(supplied) == value
            ) if isinstance(value, list) else supplied == value
            # Reject decoder metadata that contradicts the encoder-derived value.
            if supplied is not None and not matches:
                raise ValueError(
                    f"decoder {key} must match the encoder metadata."
                )
            decoder_config[key] = deepcopy(value)
        self.decoder = DiTDecoder(**decoder_config)
        self.supports_teacher_forcing = True

        DiTEncoderDecoder._validate_decoder_output(self)

        self._save_init_args({
            "encoder_kwargs": saved_encoder_kwargs, 
            "decoder_kwargs": saved_decoder_kwargs, 
            "build": build, 
        })

        # Materialize encoder, decoder, and classifier variables when requested.
        if self.build_:
            self.build()

    @property
    def encoder(self) -> DiTClassifier:
        """Return the inherited encoder/classifier network.

        Returns:
            DiTClassifier: ``self``.  A property is used instead of assigning
            the model to itself, which would create a recursive Keras tracking
            graph.
        """

        return self

    def get_config(self) -> dict[str, object]:
        """Serialize architecture settings and standard Keras model state.

        Returns:
            dict[str, object]: Constructor arguments plus ``name``,
            ``trainable``, ``dtype``, and ``dynamic``.  The nested encoder and
            decoder dictionaries are defensive copies supplied by
            :class:`ArgumentSaverModel`.
        """

        config = super().get_config()
        config.update({
            "name": self.name,
            "trainable": self.trainable,
            "dtype": self.dtype_policy.name,
            "dynamic": self.dynamic,
        })

        return config

    def build(
        self, 
        input_shape: tuple[tuple, tuple, tuple] | tuple[
            tuple, tuple, tuple, tuple
        ] | None = None, 
    ) -> None:
        """Build the composite with its four configured symbolic inputs.

        Args:
            input_shape (tuple | None): Accepted for the Keras build protocol
                but ignored; :meth:`build_model` derives shapes from the active
                resolution.  Three- and four-input eager calls remain valid.

        Returns:
            None: Encoder, classifier, decoder, and output-head variables are
            created and the outer Keras model is marked built.
        """

        symbolic_shapes = self.build_model(call_model=True)
        tf.keras.Model.build(self, symbolic_shapes)

    def build_model(self, call_model: bool = True) -> list[tf.TensorShape]:
        """Create the four-input encoder/teacher-forcing symbolic interface.

        Args:
            call_model (bool): Populate ``outputs`` through :meth:`call` when
                true; otherwise create only the symbolic inputs.

        Returns:
            list[tf.TensorShape]: Shapes for encoder images
            ``[None,H,H,C]``, timesteps ``[None]``, labels ``[None]``, and
            decoder images ``[None,H,H,C]``. Encoder and decoder patch/token
            settings may differ, but the wrapper-facing image contract is
            shared.
        """

        noisy_images = layers.Input(
            shape=(
                self.current_resolution,
                self.current_resolution,
                self.channels,
            ),
            dtype=tf.float32,
            name="noisy_images",
        )
        times = layers.Input(
            shape=(),
            dtype=tf.int32,
            name="timesteps",
        )
        labels = layers.Input(
            shape=(),
            dtype=tf.uint8,
            name="labels",
        )
        decoder_images = layers.Input(
            shape=(
                self.decoder.current_resolution,
                self.decoder.current_resolution,
                self.decoder.channels,
            ),
            dtype=tf.float32,
            name="decoder_images",
        )

        self.inputs = (noisy_images, times, labels, decoder_images)
        self.outputs = self.call(self.inputs) if call_model else None
        return [input_layer.shape for input_layer in self.inputs]

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor] | tuple[
            tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
        ], 
        full_return: bool = False, 
        training: bool | None = None, 
        min_depth: int = 0, 
    ) -> tf.Tensor | dict[str, object]:
        """Predict decoder noise and encoder-derived class probabilities.

        Args:
            inputs (tuple[tf.Tensor, ...]): ``(encoder_images, times, labels)``
                or ``(encoder_images, times, labels, decoder_images)``.
                Encoder images are ``[B,H_e,W_e,C_e]`` and explicit decoder
                images are ``[B,H_d,W_d,C_d]``; times and labels are ``[B]``.
                In the three-input form, ``encoder_images`` also feeds the
                decoder and therefore must satisfy both image interfaces.
            full_return (bool): Include the established transformer,
                classifier, and decoder intermediate fields.
            training (bool | None): Keras training mode for all branches.
            min_depth (int): First encoder depth to execute. With three inputs,
                values above zero initialize the decoder from a zero image.

        Returns:
            tf.Tensor | dict[str, object]: At ``min_depth=0``, contains decoder
            ``"noises"`` and classifier ``"classes"``. A non-full resumed
            call returns only the decoder noise tensor for inherited VAE
            sampling. Full return additionally contains
            ``cond``, ``features_list``, ``regs_list``, ``z_vals``, all four
            ``clf_*`` fields, and the decoder's ``decoder_cond``,
            ``decoder_features_list``, ``encoder_cond``, and
            ``encoder_features_list`` fields.

        Raises:
            ValueError: If ``inputs`` does not contain three or four tensors,
                or decoder routing selects a skipped encoder feature.
        """
        noisy_images, times, labels, decoder_images = \
            DiTEncoderDecoder._split_encoder_decoder_inputs(
                self, inputs, min_depth
            )

        _, encoder_cond, encoder_features, encoder_regs, encoder_z = self.encode(
            (noisy_images, times, labels), 
            min_depth=min_depth, 
            training=training, 
        )
        DiTEncoderDecoder._validate_decoder_features(self, encoder_features)
        decoder_outputs = self.decoder(
            (decoder_images, times, labels), 
            encoder_cond, 
            encoder_features, 
            full_return=True, 
            training=training, 
        )
        noises = decoder_outputs["noises"]
        # Skip classifier recomputation for a resumed noise-only call.
        if min_depth > 0 and not full_return:
            return noises
        classes, clf_cond, clf_features, clf_regs, clf_z = self.compute_class(
            encoder_features, 
            noises, 
            times, 
            labels, 
            training=training, 
        )

        outputs = {
            "noises": noises, 
            "classes": classes, 
        }
        # Attach encoder, decoder, and classifier metadata only for full returns.
        if full_return:
            outputs.update({
                "cond": encoder_cond, 
                "features_list": encoder_features, 
                "regs_list": encoder_regs, 
                "z_vals": encoder_z, 
                "clf_cond": clf_cond, 
                "clf_features_list": clf_features, 
                "clf_regs_list": clf_regs, 
                "clf_z_vals": clf_z, 
                "decoder_cond": decoder_outputs["decoder_cond"], 
                "decoder_features_list": decoder_outputs[
                    "decoder_features_list"
                ], 
                "encoder_cond": encoder_cond, 
                "encoder_features_list": encoder_features, 
            })
        return outputs

    def predict_noise(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor] | tuple[
            tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
        ], 
        full_return: bool = False, 
        training: bool | None = None, 
        min_depth: int = 0, 
    ) -> tf.Tensor | tuple[
        tf.Tensor, 
        tf.Tensor | None, 
        list[tf.Tensor | None], 
        list[tf.Tensor | None], 
        tuple[tf.Tensor | None, tf.Tensor | None], 
    ]:
        """Run the encoder-context and decoder noise branches only.

        Args:
            inputs (tuple[tf.Tensor, ...]): Three- or four-tensor input with the
                same normalization as :meth:`call`.
            full_return (bool): Return the base-compatible five-item tuple
                ``(noises, cond, features_list, regs_list, z_vals)``.
            training (bool | None): Keras training mode.
            min_depth (int): First encoder stage. With three inputs, values
                above zero initialize the decoder from a zero image.

        Returns:
            tf.Tensor | tuple: Decoder image/noise ``[B,H,W,C]``, optionally
            followed by the encoder's condition, features, regularizers, and latent stats.
            Conditionless configurations and absent/skipped intermediates use
            ``None``.

        Raises:
            ValueError: If ``inputs`` does not contain three or four tensors,
                or decoder routing selects a skipped encoder feature.
        """

        noisy_images, times, labels, decoder_images = \
            DiTEncoderDecoder._split_encoder_decoder_inputs(
                self, inputs, min_depth
            )

        _, encoder_cond, encoder_features, encoder_regs, encoder_z = self.encode(
            (noisy_images, times, labels), 
            min_depth=min_depth, 
            training=training, 
        )
        DiTEncoderDecoder._validate_decoder_features(self, encoder_features)
        decoder_outputs = self.decoder(
            (decoder_images, times, labels), 
            encoder_cond, 
            encoder_features, 
            full_return=False, 
            training=training, 
        )
        noises = decoder_outputs["noises"]

        # Preserve the extended classifier output tuple only for full returns.
        if full_return:
            return (
                noises, 
                encoder_cond, 
                encoder_features, 
                encoder_regs, 
                encoder_z, 
            )
        return noises

    def _apply_depths(
        self, 
        depth_spec: str | tuple | set | dict | list | None, 
    ) -> dict[str, dict[str, int]]:
        """Apply a validated three-branch growth specification.

        Ordinary specifications grow the encoder. Targeted dictionaries accept
        ``network``, ``classifier``, and ``decoder``. The first two retain
        :class:`DiTClassifier` semantics; the decoder value uses
        :class:`DiTDecoder` progressive layer names.

        Args:
            depth_spec (str | tuple | set | dict | list | None): Unscoped or
                targeted progressive specification.

        Returns:
            dict[str, dict[str, int]]: Growth counts for all three branches.

        Raises:
            ValueError: If a targeted dictionary contains another key.
        """

        targeted = isinstance(depth_spec, dict) and any(
            key in depth_spec for key in ("network", "classifier", "decoder")
        )
        # Split targeted growth across encoder, classifier, and decoder branches.
        if targeted:
            # Reject targeted keys outside the three architecture branches.
            if not all(
                key in ("network", "classifier", "decoder")
                for key in depth_spec
            ):
                raise ValueError(
                    "targeted depth_spec keys must be 'network', "
                    "'classifier', or 'decoder'."
                )
            classifier_spec = {
                "network": depth_spec.get("network", []),
                "classifier": depth_spec.get("classifier", []),
            }
            decoder_spec = depth_spec.get("decoder", [])
        # Treat an unscoped specification as classifier-only growth.
        else:
            classifier_spec = depth_spec
            decoder_spec = []

        growth = DiTClassifier.add_depths(self, classifier_spec)
        dims, grids = DiTEncoderDecoder._get_encoder_feature_metadata(self)
        self.decoder.set_encoder_feature_metadata(dims, grids)
        decoder_growth = self.decoder.add_depths(decoder_spec)["network"]
        DiTEncoderDecoder._validate_decoder_output(self)
        self.decoder_kwargs = deepcopy(self.decoder.get_config())
        self._init_config["decoder_kwargs"] = deepcopy(self.decoder_kwargs)

        return {
            "network": growth["network"],
            "classifier": growth.get("classifier", {
                "before": self.clf_depth,
                "added": 0,
                "after": self.clf_depth,
            }),
            "decoder": decoder_growth,
        }

    def add_depths(
        self, 
        depth_spec: str | tuple | set | dict | list | None, 
    ) -> dict[str, dict[str, int]]:
        """Grow encoder, classifier, and decoder branches transactionally.

        Ordinary specifications grow the encoder. Targeted mappings accept
        ``network``, ``classifier``, and ``decoder``. The full operation is
        validated on an unbuilt configuration clone before this model changes,
        preventing an invalid later branch from leaving earlier branches grown.

        Args:
            depth_spec (str | tuple | set | dict | list | None): Unscoped or
                targeted progressive specification.

        Returns:
            dict[str, dict[str, int]]: Before/added/after counts for all three
            branches.

        Raises:
            ValueError: If a target or layer name is unknown, or a branch
                violates its output topology.
        """

        probe_config = self.get_config()
        probe_config["build"] = False
        probe = DiTEncoderDecoderClassifier.from_config(probe_config)
        probe._apply_depths(deepcopy(depth_spec))

        return self._apply_depths(depth_spec)

    def add_class(self, source_network: object | None = None) -> None:
        """Grow encoder, classifier, and decoder class outputs together.

        Args:
            source_network (object | None): Optional already-expanded raw
                joint network used to initialize new EMA rows and output.

        Returns:
            None: Both label vocabularies and the classifier head grow by one.
        """

        DiTClassifier.add_class(self, source_network=source_network)
        self.decoder.add_class(
            source_network=(
                source_network.decoder if source_network is not None else None
            )
        )

    def predict_class(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor] | tuple[
            tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
        ], 
        max_encoder_num: int | None = -1, 
        full_return: bool = False, 
        training: bool | None = None, 
    ) -> tf.Tensor | tuple[
        tf.Tensor, 
        tf.Tensor | None, 
        list[tf.Tensor | None], 
        list[tf.Tensor | None], 
        tuple[tf.Tensor | None, tf.Tensor | None], 
    ]:
        """Classify encoder features or the decoder's final noise prediction.

        Args:
            inputs (tuple[tf.Tensor, ...]): Three- or four-tensor composite
                input.  The decoder image is ignored for feature-based
                classification and used when ``aggregate_from_noises=True``.
            max_encoder_num (int | None): Encoder loop stop; ``None`` uses the
                configured maximum and ``-1`` executes all stages.
            full_return (bool): Include classifier condition, features,
                regularizers, and latent statistics.
            training (bool | None): Keras training mode.

        Returns:
            tf.Tensor | tuple: Class probabilities ``[B,num_classes]`` or the
            same five-item full result as :meth:`DiTClassifier.predict_class`.

        Raises:
            ValueError: If ``inputs`` does not contain three or four tensors.
        """

        # Require the documented three- or four-tensor input form.
        if len(inputs) not in (3, 4):
            raise ValueError(
                "inputs must contain images, times, labels, and optionally "
                "decoder_images."
            )
        base_inputs = inputs[:3]
        # Reuse the ordinary classifier path when noise aggregation is disabled.
        if not self.aggregate_from_noises:
            return DiTClassifier.predict_class(
                self, 
                base_inputs, 
                max_encoder_num=max_encoder_num, 
                full_return=full_return, 
                training=training, 
            )

        max_encoder_num = (
            self.max_encoder_num
            if max_encoder_num is None
            else max_encoder_num
        )
        decoder_images = inputs[3] if len(inputs) == 4 else base_inputs[0]
        _, encoder_cond, encoder_features, _, _ = self.encode(
            base_inputs, 
            max_depth=max_encoder_num, 
            training=training, 
        )
        decoder_encoder_features = encoder_features + [None] * (
            len(self.decoder.encoder_feature_dims) - len(encoder_features)
        )
        DiTEncoderDecoder._validate_decoder_features(
            self, decoder_encoder_features
        )
        decoder_outputs = self.decoder(
            (decoder_images, base_inputs[1], base_inputs[2]), 
            encoder_cond, 
            decoder_encoder_features, 
            full_return=False, 
            training=training, 
        )
        noises = decoder_outputs["noises"]
        outputs = self.compute_class(
            encoder_features, 
            noises, 
            times=base_inputs[1], 
            labels=base_inputs[2], 
            training=training, 
        )
        # Return the metadata mapping only when the caller requested it.
        if full_return:
            return outputs
        return outputs[0]

    def set_current_resolution(self, resolution: int | None = None) -> None:
        """Synchronize the active encoder and decoder resolutions.

        Args:
            resolution (int | None): Positive size divisible by both patch
                sizes.  ``None`` restores each branch's configured image size.

        Returns:
            None: Encoder/classifier and decoder active resolutions are
            updated in place.
        """

        DiTEncoderDecoder.set_current_resolution(self, resolution)


def run_self_tests() -> dict[str, str]:
    """Run functional API tests for the encoder-decoder classifier.

    Args:
        None.

    Returns:
        dict[str, str]: ``{"DiTEncoderDecoderClassifier": "passed"}`` after
        inheritance, material three/four-input routing, decoder context,
        classifier modes, symbolic build, resolution execution, wrapper
        training, growth, gradients, and serialization pass.
    """

    tf.keras.backend.clear_session()
    tf.random.set_seed(109)


    from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
    from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2


    encoder_kwargs = {
        "num_classes": 2, 
        "use_cfg": True, 
        "timesteps": 4, 
        "image_size": 4, 
        "channels": 1, 
        "patch_size": 2, 
        "dim": 4, 
        "depth": 0, 
        "mha_num_heads": 1, 
        "vit_block_mlp_ratio": 1.0, 
        "feature_aggregation_ids_dict": {1: (-1,)}, 
        "clf_connection_ids_dict": {-1: (-1,)}, 
    }
    decoder_kwargs = {
        "depth": 0, 
        "shift_inputs": False, 
        "use_unpatchify": True, 
    }

    assert issubclass(DiTEncoderDecoderClassifier, DiTEncoderDecoder)
    assert issubclass(DiTEncoderDecoderClassifier, DiTClassifier)
    assert DiTEncoderDecoderClassifier.__mro__[1:3] == (
        DiTEncoderDecoder, DiTClassifier
    )

    model = DiTEncoderDecoderClassifier(
        encoder_kwargs=encoder_kwargs, 
        decoder_kwargs=decoder_kwargs, 
        name="encoder_decoder_classifier", 
    )
    assert model.encoder is model
    assert isinstance(model.decoder, DiTDecoder)
    assert model.supports_teacher_forcing is True
    assert model.use_unpatchify is False
    assert model.built and model.decoder.built
    assert model.decoder.build_ is False
    assert model.image_size == model.decoder.image_size == 4
    assert model.timesteps == model.decoder.timesteps == 4
    assert model.num_classes == model.decoder.num_classes == 2
    assert model.decoder.encoder_output_grid_size == 2
    assert model.decoder.encoder_output_dim == 4
    assert model.name_prefix == "encoder_model/"
    assert model.decoder.name_prefix == "decoder_model/"

    public_apis = {
        "build", "build_model", "call", "set_current_resolution", 
        "embed_conditions", "embed_inputs", "prepend_cls_token", 
        "slice_and_flatten_tokens", "encode", "add_depths", 
        "get_variables_names", "set_max_encoder_num", "predict_noise", 
        "compute_class", "predict_class", "get_config", "from_config", 
    }
    assert all(callable(getattr(model, name)) for name in public_apis)

    images = tf.reshape(tf.linspace(-1.0, 1.0, 32), (2, 4, 4, 1))
    decoder_images = tf.reverse(images, axis=[1])
    times = tf.constant([0, 3], dtype=tf.int32)
    labels = tf.constant([1, 2], dtype=tf.uint8)
    three_inputs = (images, times, labels)
    four_inputs = (images, times, labels, decoder_images)

    three_outputs = model(three_inputs, training=False)
    explicit_fallback = model(
        (images, times, labels, images), 
        training=False, 
    )
    tf.debugging.assert_near(
        three_outputs["noises"], explicit_fallback["noises"]
    )
    tf.debugging.assert_near(
        three_outputs["classes"], explicit_fallback["classes"]
    )
    assert set(three_outputs) == {"noises", "classes"}
    assert three_outputs["noises"].shape == (2, 4, 4, 1)
    assert three_outputs["classes"].shape == (2, 2)
    tf.debugging.assert_near(
        tf.reduce_sum(three_outputs["classes"], axis=-1), 
        tf.ones((2,)), 
    )

    full_outputs = model(
        four_inputs, 
        full_return=True, 
        training=False, 
    )
    assert set(full_outputs) == {
        "noises", "classes", "cond", "features_list", "regs_list", 
        "z_vals", "clf_cond", "clf_features_list", "clf_regs_list", 
        "clf_z_vals", "decoder_cond", "decoder_features_list", 
        "encoder_cond", "encoder_features_list", 
    }
    assert full_outputs["noises"].shape == (2, 4, 4, 1)
    assert full_outputs["classes"].shape == (2, 2)
    assert full_outputs["encoder_cond"] is full_outputs["cond"]
    assert full_outputs["encoder_features_list"] is full_outputs["features_list"]
    assert len(full_outputs["features_list"]) == model.depth + 1
    assert len(full_outputs["decoder_features_list"]) == model.decoder.depth + 1

    encoded = model.encode(three_inputs, training=False)
    assert len(encoded) == 5
    assert encoded[0].shape == (2, 4, 4)
    decoder_tokens, decoder_cond, decoder_features = model.decoder.decode(
        (decoder_images, times, labels), 
        encoded[1], 
        encoded[2], 
        training=False, 
    )
    direct_noises = model.decoder.unpatchifier(
        (decoder_tokens, decoder_cond), 
        training=False, 
    )
    tf.debugging.assert_near(full_outputs["noises"], direct_noises)
    assert decoder_features[-1].shape == decoder_tokens.shape
    direct_decoder_outputs = model.decoder(
        (decoder_images, times, labels), 
        encoded[1], 
        encoded[2], 
        full_return=True, 
        training=False, 
    )
    tf.debugging.assert_near(
        direct_decoder_outputs["noises"], 
        full_outputs["noises"], 
    )

    predicted_noise = model.predict_noise(four_inputs, training=False)
    tf.debugging.assert_near(predicted_noise, full_outputs["noises"])
    noise_details = model.predict_noise(
        four_inputs, 
        full_return=True, 
        training=False, 
    )
    assert len(noise_details) == 5
    tf.debugging.assert_near(noise_details[0], predicted_noise)
    assert noise_details[1] is not None and len(noise_details[2]) == 1

    predicted_classes = model.predict_class(four_inputs, training=False)
    tf.debugging.assert_near(predicted_classes, full_outputs["classes"])
    class_details = model.predict_class(
        four_inputs, 
        full_return=True, 
        training=False, 
    )
    assert len(class_details) == 5
    computed_classes = model.compute_class(
        encoded[2], 
        direct_noises, 
        times, 
        labels, 
        training=False, 
    )
    tf.debugging.assert_near(computed_classes[0], predicted_classes)

    decoder_head_kernels = [
        variable for variable in model.decoder.trainable_variables
        if "unpatchifier/ffn/kernel" in variable.name
    ]
    assert len(decoder_head_kernels) == 1
    decoder_head_kernel = decoder_head_kernels[0]
    decoder_head_kernel.assign(tf.reshape(
        tf.linspace(-0.5, 0.5, tf.size(decoder_head_kernel)), 
        decoder_head_kernel.shape, 
    ))
    same_decoder_outputs = model(
        (images, times, labels, images), training=False
    )
    distinct_decoder_outputs = model(four_inputs, training=False)
    assert float(tf.reduce_max(tf.abs(
        same_decoder_outputs["noises"]
        - distinct_decoder_outputs["noises"]
    ))) > 1e-4
    tf.debugging.assert_near(
        same_decoder_outputs["classes"], 
        distinct_decoder_outputs["classes"], 
    )
    tf.debugging.assert_near(
        model.predict_class(
            (images, times, labels, images), training=False
        ), 
        model.predict_class(four_inputs, training=False), 
    )

    cond, time_embeds, label_embeds = model.embed_conditions(
        times, 
        labels, 
        model.cond_type, 
        full_return=True, 
        training=False, 
    )
    assert cond.shape == time_embeds.shape == label_embeds.shape == (2, 4)
    embedded, embedded_cond = model.embed_inputs(
        three_inputs, 
        model.cond_type, 
        training=False, 
    )
    assert embedded.shape == (2, 4, 4)
    tf.debugging.assert_near(embedded_cond, cond)
    assert model.slice_and_flatten_tokens(embedded, 0, 1).shape == (2, 4)

    model.set_max_encoder_num(-1)
    assert model.max_encoder_num == -1
    model.set_max_encoder_num()
    assert isinstance(model.max_encoder_num, int)
    variable_names = model.get_variables_names()
    assert len(variable_names) == len(model.trainable_variables)
    assert not any("encoder_model/unpatchifier" in name for name in variable_names)
    decoder_variable_ids = {id(value) for value in model.decoder.trainable_variables}
    assert decoder_variable_ids
    assert any(id(value) in decoder_variable_ids for value in model.trainable_variables)

    with tf.GradientTape() as tape:
        training_outputs = model(four_inputs, training=True)
        objective = (
            tf.reduce_sum(training_outputs["noises"])
            + tf.reduce_sum(training_outputs["classes"])
        )
    gradients = tape.gradient(objective, model.trainable_variables)
    gradient_by_id = {
        id(variable): gradient
        for variable, gradient in zip(model.trainable_variables, gradients)
    }
    assert any(
        gradient_by_id[id(variable)] is not None
        for variable in model.decoder.trainable_variables
    )
    assert any(
        gradient is not None
        for variable, gradient in zip(model.trainable_variables, gradients)
        if id(variable) not in decoder_variable_ids
    )

    model.set_current_resolution(8)
    assert model.current_resolution == model.decoder.current_resolution == 8
    resized_images = tf.image.resize(images, (8, 8), method="nearest")
    resized_outputs = model((
        resized_images, 
        times, 
        labels, 
        tf.reverse(resized_images, axis=[1]), 
    ), training=False)
    assert resized_outputs["noises"].shape == (2, 8, 8, 1)
    assert resized_outputs["classes"].shape == (2, 2)
    model.set_current_resolution()
    assert model.current_resolution == model.decoder.current_resolution == 4

    config = model.get_config()
    assert config["encoder_kwargs"] == encoder_kwargs
    assert config["decoder_kwargs"] == decoder_kwargs
    assert config["build"] is True
    clone = DiTEncoderDecoderClassifier.from_config(config)
    clone_outputs = clone(four_inputs, training=False)
    assert clone_outputs["noises"].shape == (2, 4, 4, 1)
    assert clone_outputs["classes"].shape == (2, 2)
    assert len(clone.weights) == len(model.weights)
    assert clone.name == model.name
    assert clone.dtype_policy.name == model.dtype_policy.name
    assert clone.trainable is model.trainable

    keras_state_model = DiTEncoderDecoderClassifier(
        encoder_kwargs=encoder_kwargs, 
        decoder_kwargs=decoder_kwargs, 
        build=False, 
        name="serialized_encoder_decoder_classifier", 
        trainable=False, 
        dtype="float64", 
    )
    keras_state_clone = DiTEncoderDecoderClassifier.from_config(
        keras_state_model.get_config()
    )
    assert keras_state_clone.name == "serialized_encoder_decoder_classifier"
    assert keras_state_clone.trainable is False
    assert keras_state_clone.dtype_policy.name == "float64"
    assert keras_state_clone.decoder.dtype_policy.name == "float64"
    assert keras_state_clone.dynamic is keras_state_model.dynamic

    wide_condition_model = DiTEncoderDecoderClassifier(
        encoder_kwargs={
            **encoder_kwargs, 
            "cond_dim": 6, 
        }, 
        decoder_kwargs={
            **decoder_kwargs, 
            "depth": 1, 
            "cond_dim": 8, 
            "decoder_separate_cond": True, 
            "mha_num_heads": 1, 
            "vit_block_mlp_ratio": 1.0, 
        }, 
    )
    wide_condition_outputs = wide_condition_model(
        four_inputs, 
        full_return=True, 
        training=False, 
    )
    assert wide_condition_model.decoder.cond_dim == 8
    assert wide_condition_outputs["cond"].shape == (2, 6)
    assert wide_condition_outputs["decoder_cond"].shape == (2, 8)
    assert len(wide_condition_outputs["decoder_features_list"]) == 2

    conditionless_model = DiTEncoderDecoderClassifier(
        encoder_kwargs={
            **encoder_kwargs, 
            "cond_type": None, 
            "ln_no_adaptation": True, 
        }, 
        decoder_kwargs={
            **decoder_kwargs, 
            "ln_no_adaptation": True, 
        }, 
    )
    conditionless_outputs = conditionless_model(
        four_inputs, 
        full_return=True, 
        training=False, 
    )
    assert conditionless_outputs["cond"] is None
    assert conditionless_outputs["decoder_cond"].shape == (2, 4)
    assert conditionless_outputs["noises"].shape == (2, 4, 4, 1)

    deferred = DiTEncoderDecoderClassifier(
        decoder_kwargs=decoder_kwargs,
        build=False,
        **encoder_kwargs,
    )
    assert deferred.built is False
    deferred_shapes = deferred.build_model(call_model=False)
    assert deferred_shapes == [
        tf.TensorShape([None, 4, 4, 1]), 
        tf.TensorShape([None]), 
        tf.TensorShape([None]), 
        tf.TensorShape([None, 4, 4, 1]), 
    ]
    assert deferred.outputs is None
    deferred.build()
    assert deferred.built
    assert deferred(three_inputs)["noises"].shape == (2, 4, 4, 1)

    growth = model.add_depths({
        "network": "vision_transformer_block", 
        "classifier": [], 
    })
    assert growth["network"] == {"before": 0, "added": 1, "after": 1}
    assert growth["classifier"] == {"before": 1, "added": 0, "after": 1}
    assert model.depth == 1 and model.decoder.depth == 0
    assert model(four_inputs)["noises"].shape == (2, 4, 4, 1)

    aggregate_model = DiTEncoderDecoderClassifier(
        encoder_kwargs={
            **encoder_kwargs, 
            "aggregate_from_noises": True, 
        }, 
        decoder_kwargs=decoder_kwargs, 
    )
    assert aggregate_model.use_unpatchify is False
    assert not any(
        "encoder_model/unpatchifier" in variable.name
        for variable in aggregate_model.trainable_variables
    )
    aggregate_head_kernels = [
        variable for variable in aggregate_model.decoder.trainable_variables
        if "unpatchifier/ffn/kernel" in variable.name
    ]
    assert len(aggregate_head_kernels) == 1
    aggregate_head_kernel = aggregate_head_kernels[0]
    aggregate_head_kernel.assign(tf.reshape(
        tf.linspace(-0.5, 0.5, tf.size(aggregate_head_kernel)), 
        aggregate_head_kernel.shape, 
    ))
    aggregate_gate_biases = [
        variable for variable in aggregate_model.trainable_variables
        if "clf_depth_1_encoder_block/mha_layer_norm/"
           "mlp/final_layer/bias" in variable.name
    ]
    assert len(aggregate_gate_biases) == 1
    aggregate_gate_bias = aggregate_gate_biases[0]
    aggregate_gate_dim = aggregate_model.clf_layers_dicts[0][
        aggregate_model.VTB
    ].query_dim
    aggregate_gate_bias.assign(tf.concat([
        tf.zeros((aggregate_gate_bias.shape[0] - aggregate_gate_dim,)), 
        tf.ones((aggregate_gate_dim,)), 
    ], axis=0))
    aggregate_same_outputs = aggregate_model(
        (images, times, labels, images), training=False
    )
    aggregate_outputs = aggregate_model(four_inputs, training=False)
    assert float(tf.reduce_max(tf.abs(
        aggregate_same_outputs["noises"] - aggregate_outputs["noises"]
    ))) > 1e-4
    assert float(tf.reduce_max(tf.abs(
        aggregate_same_outputs["classes"] - aggregate_outputs["classes"]
    ))) > 1e-4
    tf.debugging.assert_near(
        aggregate_outputs["classes"], 
        aggregate_model.predict_class(four_inputs, training=False), 
    )
    aggregate_config = aggregate_model.get_config()
    assert aggregate_config["aggregate_from_noises"] is True
    assert aggregate_config["use_unpatchify"] is False
    aggregate_clone = DiTEncoderDecoderClassifier.from_config(
        aggregate_config
    )
    assert aggregate_clone.aggregate_from_noises is True
    assert aggregate_clone.use_unpatchify is False
    assert not any(
        "encoder_model/unpatchifier" in variable.name
        for variable in aggregate_clone.trainable_variables
    )
    assert aggregate_clone(four_inputs)["classes"].shape == (2, 2)
    try:
        DiTEncoderDecoderClassifier(
            encoder_kwargs={
                **encoder_kwargs, 
                "aggregate_from_noises": True, 
            }, 
            decoder_kwargs={
                **decoder_kwargs, 
                "use_unpatchify": False, 
            }, 
            build=False, 
        )
    except ValueError as error:
        assert "decoder use_unpatchify=True" in str(error)
    else:
        raise AssertionError(
            "Noise-based classification must require decoder unpatchification"
        )
    for incompatible_decoder_kwargs in (
        {**decoder_kwargs, "image_size": 8}, 
        {**decoder_kwargs, "channels": 2}, 
    ):
        try:
            DiTEncoderDecoderClassifier(
                encoder_kwargs={
                    **encoder_kwargs, 
                    "aggregate_from_noises": True, 
                }, 
                decoder_kwargs=incompatible_decoder_kwargs, 
                build=False, 
            )
        except ValueError as error:
            assert "encoder and decoder image_size/channels" in str(error)
        else:
            raise AssertionError(
                "Noise-based classification must require matching image metadata"
            )
    try:
        DiTEncoderDecoderClassifier(
            encoder_kwargs={
                **encoder_kwargs, 
                "aggregate_from_noises": True, 
            }, 
            decoder_kwargs={
                **decoder_kwargs, 
                "depth": 1, 
                "vit_block_ids": [], 
                "downsample_ids": [1], 
            }, 
            build=False, 
        )
    except ValueError as error:
        assert "final token grid" in str(error)
    else:
        raise AssertionError(
            "Noise-based classification must require a restored decoder grid"
        )

    shifted_token_model = DiTEncoderDecoderClassifier(
        encoder_kwargs=encoder_kwargs, 
        decoder_kwargs={
            **decoder_kwargs, 
            "shift_inputs": True, 
            "cls_token_type": "new_weight", 
        }, 
    )
    shifted_token_outputs = shifted_token_model(four_inputs, training=False)
    assert shifted_token_outputs["noises"].shape == (2, 4, 4, 1)
    assert shifted_token_outputs["classes"].shape == (2, 2)

    wrapper_network = DiTEncoderDecoderClassifier(
        encoder_kwargs=encoder_kwargs, 
        decoder_kwargs=decoder_kwargs, 
        name="wrapped_encoder_decoder_classifier", 
    )
    wrapper = DiffusionClassifier(
        network=wrapper_network, 
        use_ema=False, 
        test_network_name="raw", 
        scheduler_name="linear", 
        test_steps=2, 
        p_uncond=0.0, 
        mask_by_nulls=False, 
        seed=73, 
    )
    wrapper.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3), 
        loss="mse", 
        run_eagerly=True, 
    )
    wrapper_raw_outputs = wrapper_network(three_inputs, training=False)
    assert wrapper_raw_outputs["noises"].shape == (2, 4, 4, 1)
    assert wrapper_raw_outputs["classes"].shape == (2, 2)
    wrapper_results = wrapper.train_step((images, labels - 1))
    assert {
        "loss", "noise_loss", "classifier_loss", "classifier_accuracy"
    } == set(wrapper_results)
    assert all(
        bool(tf.math.is_finite(value)) for value in wrapper_results.values()
    )
    wrapper_progressive = wrapper.fit_progressively(
        stage_tasks="depths_only", 
        depths=[{
            "network": [], 
            "classifier": [], 
            "decoder": "vision_transformer_block", 
        }], 
        x=(images, labels - 1), 
        batch_size=2, 
        stage_epochs=0, 
        final_epochs=0, 
        stages_verbose=False, 
        verbose=0, 
    )
    assert wrapper_progressive.progressive_stages[-1][
        "depth_growth"
    ]["decoder"]["added"] == 1
    assert wrapper.network.decoder.depth == 1
    assert "noise_loss" in wrapper.train_step((images, labels - 1))

    v2_network = DiTEncoderDecoderClassifier(
        encoder_kwargs=encoder_kwargs, 
        decoder_kwargs={
            **decoder_kwargs, 
            "depth": 1, 
            "mha_num_heads": 1, 
            "vit_block_mlp_ratio": 1.0, 
        }, 
    )
    v2_wrapper = DiffusionClassifierV2(
        network=v2_network, 
        use_ema=False, 
        test_network_name="raw", 
        scheduler_name="linear", 
        test_steps=2, 
        p_uncond=0.0, 
        mask_by_nulls=False, 
        seed=79, 
    )
    v2_wrapper.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3), 
        loss="mse", 
        run_eagerly=True, 
    )
    generator_results = v2_wrapper.generator_train_step((images, labels - 1))
    classifier_results = v2_wrapper.discriminator_train_step(
        (images, labels - 1)
    )
    assert "noise_loss" in generator_results
    assert {"classifier_loss", "classifier_accuracy"} <= set(
        classifier_results
    )
    assert len(v2_wrapper.gen_trainable_variables) + len(
        v2_wrapper.clf_trainable_variables
    ) == len(v2_network.trainable_variables)
    v2_progressive = v2_wrapper.fit_progressively(
        stage_tasks="depths_only", 
        depths=[{
            "network": [],
            "classifier": [],
            "decoder": "vision_transformer_block",
        }], 
        x=(images, labels - 1), 
        batch_size=2, 
        stage_epochs=0, 
        final_epochs=0, 
        stages_verbose=False, 
        verbose=0, 
    )
    assert v2_progressive.progressive_stages[-1][
        "depth_growth"
    ]["decoder"]["added"] == 1
    assert v2_network.decoder.depth == 2
    assert "noise_loss" in v2_wrapper.generator_train_step(
        (images, labels - 1)
    )
    assert len(v2_wrapper.gen_trainable_variables) + len(
        v2_wrapper.clf_trainable_variables
    ) == len(v2_network.trainable_variables)
    for source_wrapper in (wrapper, v2_wrapper):
        restored_wrapper = type(source_wrapper).from_config(
            source_wrapper.get_config()
        )
        assert isinstance(
            restored_wrapper.network, DiTEncoderDecoderClassifier
        )
        assert len(restored_wrapper.network.weights) == len(
            source_wrapper.network.weights
        )
        for source, restored in zip(
            source_wrapper.network.weights, 
            restored_wrapper.network.weights, 
        ):
            tf.debugging.assert_near(source, restored)

    for malformed_inputs in (
        (images, times), 
        (images, times, labels, decoder_images, decoder_images),
    ):
        for method in (model.call, model.predict_noise, model.predict_class):
            try:
                method(malformed_inputs, training=False)
            except ValueError as error:
                assert "inputs must contain images, times, labels" in str(error)
            else:
                raise AssertionError(
                    f"{method.__name__} must reject malformed input arity"
                )

    # Fresh implicit mutable routing dictionaries must remain reusable across
    # sequential construction.
    implicit_config = {
        key: value
        for key, value in encoder_kwargs.items()
        if key not in (
            "feature_aggregation_ids_dict", 
            "clf_connection_ids_dict", 
        )
    }
    first = DiTEncoderDecoderClassifier(
        decoder_kwargs=decoder_kwargs, 
        build=False, 
        **implicit_config, 
    )
    second = DiTEncoderDecoderClassifier(
        decoder_kwargs=decoder_kwargs, 
        build=False, 
        **implicit_config, 
    )
    assert first.feature_aggregation_ids_dict == second.feature_aggregation_ids_dict
    assert first.clf_connection_ids_dict == second.clf_connection_ids_dict

    progressive = DiTEncoderDecoderClassifier(
        encoder_kwargs=encoder_kwargs, 
        decoder_kwargs=decoder_kwargs, 
        build=False, 
    )
    before_depths = (
        progressive.depth, 
        progressive.clf_depth, 
        progressive.decoder.depth, 
    )
    try:
        progressive.add_depths({
            "network": "vision_transformer_block", 
            "classifier": "vision_transformer_block", 
            "decoder": "not_a_layer", 
        })
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid branch growth must fail preflight.")
    assert (
        progressive.depth, 
        progressive.clf_depth, 
        progressive.decoder.depth, 
    ) == before_depths
    progressive_growth = progressive.add_depths({
        "network": "vision_transformer_block", 
        "classifier": "vision_transformer_block", 
        "decoder": "vision_transformer_block", 
    })
    assert progressive_growth == {
        "network": {"before": 0, "added": 1, "after": 1}, 
        "classifier": {"before": 1, "added": 1, "after": 2}, 
        "decoder": {"before": 0, "added": 1, "after": 1}, 
    }
    assert progressive.decoder.encoder_feature_dims == [4, 4]
    progressive_outputs = progressive(three_inputs, training=False)
    assert progressive_outputs["noises"].shape == (2, 4, 4, 1)
    assert progressive_outputs["classes"].shape == (2, 2)
    progressive_clone = DiTEncoderDecoderClassifier.from_config(
        progressive.get_config()
    )
    progressive_clone(three_inputs, training=False)
    progressive_clone.set_weights(progressive.get_weights())
    assert progressive_clone.depth == 1
    assert progressive_clone.clf_depth == 2
    assert progressive_clone.decoder.depth == 1
    clone_outputs = progressive_clone(three_inputs, training=False)
    tf.debugging.assert_near(
        clone_outputs["noises"], progressive_outputs["noises"]
    )

    vae_encoder_kwargs = {
        **encoder_kwargs, 
        "depth": 2, 
        "vit_block_ids": [], 
        "reshaper_ids_dict": {1: "flatten", 2: "unflatten"}, 
        "reshaper_kwargs": {"add_kl": True, "latent_dim_ratio": 0.5}, 
    }
    vae_decoder_kwargs = {
        **decoder_kwargs, 
        "depth": 1, 
        "mha_num_heads": 1, 
        "vit_block_mlp_ratio": 1.0, 
    }
    for wrapper_type in (DiffusionClassifier, DiffusionClassifierV2):
        vae_model = DiTEncoderDecoderClassifier(
            encoder_kwargs=vae_encoder_kwargs, 
            decoder_kwargs=vae_decoder_kwargs, 
        )
        vae_wrapper = wrapper_type(
            network=vae_model, 
            use_ema=False, 
            swap_noise_image=True, 
            test_steps=2, 
        )
        vae_sample = vae_wrapper.sample_vae(
            network_name="raw", labels=[1, 2]
        )
        assert vae_sample.shape == (2, 4, 4, 1)
        assert bool(tf.reduce_all(tf.math.is_finite(vae_sample)))

    tf.keras.backend.clear_session()
    return {"DiTEncoderDecoderClassifier": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
