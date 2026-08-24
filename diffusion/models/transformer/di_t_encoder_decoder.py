"""Diffusion-transformer encoder with a context-aware DiT decoder head."""

import tensorflow as tf
from tensorflow.keras import layers

from math import isqrt

from copy import deepcopy

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.transformer.di_t_decoder import DiTDecoder


class DiTEncoderDecoder(DiffusionTransformer):
    """Use a diffusion transformer as an encoder for a DiT decoder.

    The object itself owns the complete :class:`DiffusionTransformer` encoder
    state, so embedding, :meth:`encode`, progressive :meth:`add_depths`, and
    variable-inspection APIs work unchanged.  :attr:`encoder` is a read-only
    alias for ``self`` and :attr:`decoder` is a separately configured
    :class:`DiTDecoder`.  The unused encoder output head is disabled; only the
    decoder owns the final noise/image head.

    Calls accept ``(images, times, labels)`` or ``(images, times, labels,
    decoder_images)``.  The three-input form reuses ``images`` as the decoder
    input for compatibility with :class:`DiffusionModel` and sampling.  The
    four-input form supplies an explicit teacher-forcing image.  When resumed
    from an encoder bottleneck, the three-input form starts the decoder from a
    zero image, which keeps :meth:`DiffusionModel.sample_vae` compatible.
    Internally the decoder receives its distinct encoder-condition and
    encoder-feature-list arguments through its normal structured API. Decoder blocks at depths
    1..N cross-attend to the final encoder feature by default.  Decoder depth 0
    has no attention block and uses only the condition and decoder image.

    The ordinary result is the decoder's noise/image tensor, matching
    :class:`DiffusionTransformer`.  ``full_return=True`` returns
    ``(noises, encoder_cond, encoder_features, encoder_regs, encoder_z_vals)``.
    Decoder intermediates remain available from ``model.decoder``.

    Attributes:
        encoder (DiffusionTransformer): Read-only alias returning ``self``.
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
        """Initialize the encoder state and attached decoder.

        Args:
            encoder_kwargs (dict[str, object] | None): Nested
                :class:`DiffusionTransformer` arguments, including dimensions,
                condition/token choices, every routing ``*_ids_dict``, and
                component ``*_kwargs``.  Flat values in ``**kwargs`` override
                equal nested keys.  ``None`` uses transformer defaults.
                ``use_unpatchify`` is forced to false because the attached
                decoder exclusively owns the composite output head.
            decoder_kwargs (dict[str, object] | None): :class:`DiTDecoder`
                arguments.  Shared class, schedule, image, patch, token-width,
                and condition-width values default to the encoder.  Encoder
                feature dimensions and grids are derived from the actual
                encoder; explicitly supplied metadata must match.  A
                supplied ``build`` value is ignored because this model owns
                symbolic construction.  ``shift_inputs`` defaults to ``False``;
                pass ``True`` for right-shifted teacher forcing.  The decoder
                also inherits the outer dtype policy unless this mapping
                provides its own ``dtype``. Its image size and channels must
                match the encoder, and ``use_unpatchify`` must be true so the
                generic diffusion wrapper receives image-shaped predictions.
                ``cond_dim`` must also match unless
                ``decoder_separate_cond=True``. Any decoder timestep/label
                embedding tables must cover the encoder's wrapper-visible ID
                ranges. Feature-width merges and encoder features used as
                cross-attention queries require matching encoder/decoder
                class-token presence; attention values may differ in length.
                Configure KL bottlenecks and token regularizers on the encoder,
                because generic wrapper losses read the encoder metadata.
            build (bool): Build the four-input symbolic graph immediately.
                ``False`` defers building until :meth:`build` or the first
                Keras call.
            **kwargs (object): Flat transformer arguments plus standard Keras
                ``Model`` options ``name``, ``trainable``, ``dtype``, and
                ``dynamic``.  Flat values take precedence over
                ``encoder_kwargs``.

        Returns:
            None: Encoder layers, decoder layers, serialization state, and
            optionally the symbolic graph are initialized.

        Raises:
            ValueError: If decoder encoder-feature metadata contradicts the
                constructed encoder, or its output cannot match the generic
                diffusion wrapper's image contract.
        """

        saved_encoder_kwargs = (
            {} if encoder_kwargs is None else deepcopy(encoder_kwargs)
        )
        saved_decoder_kwargs = (
            {} if decoder_kwargs is None else deepcopy(decoder_kwargs)
        )

        effective_encoder_kwargs = deepcopy(saved_encoder_kwargs)
        effective_encoder_kwargs.update(deepcopy(kwargs))
        effective_encoder_kwargs["build"] = False
        effective_encoder_kwargs["use_unpatchify"] = False
        effective_encoder_kwargs.setdefault("name_prefix", "encoder_model/")
        DiffusionTransformer.__init__(self, **effective_encoder_kwargs)

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
            self._get_encoder_feature_metadata()
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
        self._validate_decoder_output()

        self.supports_teacher_forcing = True

        self._save_init_args({
            "encoder_kwargs": saved_encoder_kwargs, 
            "decoder_kwargs": saved_decoder_kwargs, 
            "build": build, 
        })

        # Materialize the combined encoder-decoder variables when requested.
        if self.build_:
            self.build()

    @property
    def encoder(self) -> DiffusionTransformer:
        """Return the transformer encoder represented by this object.

        Returns:
            DiffusionTransformer: ``self``.  Avoiding a self-assignment keeps
            Keras layer tracking acyclic.
        """

        return self

    def _get_encoder_feature_metadata(
        self, 
    ) -> tuple[list[int], list[int | None]]:
        """Describe every depth-indexed encoder feature for the decoder.

        Returns:
            tuple[list[int], list[int | None]]: Feature widths and matching
            spatial grid sides for depth 0 through ``self.depth``.
        """

        dims = [self.dim]
        grids = [self.grid_size]
        for index in range(self.depth):
            dims.append(self._get_last_output_dim(
                index, 
                self.layers_dicts, 
                self.dim
            ))
            grids.append(self._get_last_grid_size(
                index, 
                self.layers_dicts, 
                self.grid_size
            ))

        return dims, grids

    def _get_last_grid_size(
        self, 
        i: int, 
        layers_dicts: list[dict], 
        base_grid_size: int, 
        skip_reshaper: bool = False
    ) -> int | None:
        """Resolve spatial state without losing explicit flat representations.

        Args:
            i (int): Zero-based stage index, or ``-1`` for depth zero.
            layers_dicts (list[dict]): Encoder stage dictionaries.
            base_grid_size (int): Depth-zero patch grid side.
            skip_reshaper (bool): Ignore reshaper changes when true.

        Returns:
            int | None: Latest square grid side, or ``None`` after a flatten
            reshaper until a later unflatten/spatial layer restores a grid.
        """

        # Return the patch grid before any encoder depth executes.
        if i == -1:
            return base_grid_size

        grid_size = None
        grid_was_set = False
        # Inspect the requested encoder stage when it exists.
        if i < len(layers_dicts):
            stage = layers_dicts[i]
            for key in (self.LM, self.DS, self.US):
                # Let spatial mixers and scalers update the encoder grid.
                if key in stage:
                    grid_size = stage[key].output_grid_size
                    grid_was_set = True

            # Derive spatial or flat metadata from a non-skipped reshaper.
            if self.R in stage and not skip_reshaper:
                output_shape = stage[self.R].output_shape[0]
                grid_size = isqrt(output_shape[1]) \
                    if len(output_shape) == 3 else None
                grid_was_set = True

        return grid_size if grid_was_set else self._get_last_grid_size(
            i - 1, 
            layers_dicts, 
            base_grid_size, 
            skip_reshaper
        )

    def _validate_decoder_output(self) -> None:
        """Validate the generic diffusion wrapper's image contract.

        Returns:
            None: A compatible decoder is left unchanged.

        Raises:
            ValueError: If the decoder emits tokens, has incompatible shared
                conditioning or embedding-table capacity, reconstructs a
                different image size/channel count, or requests auxiliary
                losses that generic wrappers read only from the encoder.
        """

        final_grid = self.decoder._get_last_grid_size(
            self.decoder.depth - 1, 
            self.decoder.layers_dicts, 
            self.decoder.grid_size, 
        )
        # Require the decoder to reconstruct image-shaped noise predictions.
        if not self.decoder.use_unpatchify:
            raise ValueError(
                "DiTEncoderDecoder requires decoder use_unpatchify=True."
            )
        # Keep encoder and decoder image geometry identical.
        if self.decoder.image_size != self.image_size or \
        self.decoder.channels != self.channels:
            raise ValueError(
                "encoder and decoder image_size/channels must match."
            )
        # Shared conditioning requires matching encoder and decoder widths.
        if not self.decoder.decoder_separate_cond and \
        self.cond_type is not None and self.decoder.cond_dim != self.cond_dim:
            raise ValueError(
                "a decoder that reuses encoder conditioning must have the "
                "same cond_dim."
            )
        # Ensure a decoder-owned time embedding covers every encoder timestep.
        if self.decoder.time_embedder is not None and \
        self.decoder.timesteps < self.timesteps:
            raise ValueError(
                "decoder timestep embeddings must cover encoder timesteps."
            )
        # Ensure a decoder-owned label embedding covers every encoder label ID.
        if self.decoder.label_embedder is not None and \
        self.decoder.num_labels < self.num_labels:
            raise ValueError(
                "decoder label embeddings must cover encoder labels."
            )
        # Detect feature-width routes made unsafe by mismatched class-token presence.
        if bool(self.cls_token_type) != bool(self.decoder.cls_token_type):
            feature_routes_require_match = any(
                self._handler_merges_feature_width(
                    self.decoder.connection_kwargs
                    if depth in self.decoder.connection_ids_dict
                    else self.decoder.feature_aggregation_kwargs
                )
                for depth in self.decoder.feature_aggregation_ids_dict
            )
            query_routes_require_match = (
                self.decoder.cross_attention_plug_type == "queries"
                and any(
                    depth in self.decoder.vit_block_ids
                    for depth in
                    self.decoder.cross_attention_aggregation_ids_dict
                )
            )
            cross_connector_requires_match = any(
                depth in self.decoder.cross_attention_ids_dict
                and self._handler_merges_feature_width(
                    self.decoder.cross_attention_kwargs
                )
                for depth in
                self.decoder.cross_attention_aggregation_ids_dict
            )
            # Reject routed encoder/decoder widths that cannot be merged safely.
            if feature_routes_require_match or query_routes_require_match or \
            cross_connector_requires_match:
                raise ValueError(
                    "encoder and decoder class-token settings must match for "
                    "feature-width merges and cross-attention query routes."
                )
        # Require the decoder's terminal token grid to match the image head.
        if final_grid is None or \
        final_grid * self.decoder.patch_size != self.image_size:
            raise ValueError(
                "the decoder's final token grid must reconstruct image_size."
            )
        # Reject decoder KL outputs that the wrapper API cannot expose.
        if self.decoder.reshaper_kwargs.get("add_kl", False):
            raise ValueError(
                "decoder KL bottlenecks are not exposed by DiffusionModel; "
                "configure KL reshaping on the encoder."
            )
        # Reject decoder token regularizers that the wrapper API cannot expose.
        if self.decoder.cls_token_regularizer_ids:
            raise ValueError(
                "decoder token regularizers are not exposed by DiffusionModel; "
                "configure token regularizers on the encoder."
            )

    @staticmethod
    def _handler_merges_feature_width(handler_kwargs: dict) -> bool:
        """Return whether a rank-3 handler requires equal token counts.

        Args:
            handler_kwargs (dict[str, object]): ``FeatureHandler`` merge
                options containing ``connect_type`` and ``connect_axis``.

        Returns:
            bool: True for addition or channel-axis concatenation; false for
            token-axis concatenation.
        """

        return handler_kwargs.get("connect_type", "concat") != "concat" or \
            handler_kwargs.get("connect_axis", -1) in (-1, 2)

    def get_config(self) -> dict[str, object]:
        """Serialize architecture settings and standard Keras model state.

        Returns:
            dict[str, object]: Saved transformer/decoder constructor values plus
            ``name``, ``trainable``, ``dtype``, and ``dynamic``.
        """

        config = super().get_config()
        config.update({
            "name": self.name,
            "trainable": self.trainable,
            "dtype": self.dtype_policy.name,
            "dynamic": self.dynamic,
        })

        return config

    def __deepcopy__(self, memo: dict[int, object]) -> "DiTEncoderDecoder":
        """Copy the composite through its public configuration and weights.

        Generic diffusion wrappers deep-copy their configured network during
        reconstruction and EMA creation. Rebuilding explicitly avoids Keras'
        transient in-memory SavedModel copy while preserving the same raw
        architecture and current weights.

        Args:
            memo (dict[int, object]): Standard ``copy.deepcopy`` memo mapping.

        Returns:
            DiTEncoderDecoder: Independent same-type network clone.
        """
        # Reuse an existing clone to preserve deepcopy memo semantics.
        if id(self) in memo:
            return memo[id(self)]

        clone = type(self).from_config(self.get_config())
        memo[id(self)] = clone
        # Copy learned weights only after the clone has compatible variables.
        if self.weights:
            # Build a deferred clone before assigning weights.
            if not clone.weights:
                clone.build()
            clone.set_weights(self.get_weights())

        return clone

    def build(
        self, 
        input_shape: tuple[tuple, tuple, tuple] | tuple[
            tuple, tuple, tuple, tuple
        ] | None = None, 
    ) -> None:
        """Build the composite against its four symbolic inputs.

        Args:
            input_shape (tuple | None): Accepted by the Keras build protocol but
                ignored; active encoder and decoder resolutions determine the
                symbolic shapes.

        Returns:
            None: Encoder/decoder variables are created and the outer model is
            marked built.
        """

        symbolic_shapes = self.build_model(call_model=True)
        tf.keras.Model.build(self, symbolic_shapes)

    def build_model(self, call_model: bool = True) -> list[tf.TensorShape]:
        """Create encoder and teacher-forcing symbolic inputs.

        Args:
            call_model (bool): Populate ``outputs`` through :meth:`call` when
                true; false creates only the four symbolic inputs.

        Returns:
            list[tf.TensorShape]: Encoder image ``[None,H,H,C]``, timestep and
            label vectors ``[None]``, and decoder image
            ``[None,decoder_H,decoder_H,decoder_C]``.
        """

        encoder_images = layers.Input(
            shape=(
                self.current_resolution, 
                self.current_resolution, 
                self.channels, 
            ),
            dtype=tf.float32, 
            name="encoder_images", 
        )
        times = layers.Input(shape=(), dtype=tf.int32, name="timesteps")
        labels = layers.Input(shape=(), dtype=tf.uint8, name="labels")
        decoder_images = layers.Input(
            shape=(
                self.decoder.current_resolution, 
                self.decoder.current_resolution, 
                self.decoder.channels, 
            ),
            dtype=tf.float32, 
            name="decoder_images", 
        )

        self.inputs = (encoder_images, times, labels, decoder_images)
        self.outputs = self.call(self.inputs) if call_model else None

        return [input_layer.shape for input_layer in self.inputs]

    def _split_encoder_decoder_inputs(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor] | tuple[
            tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
        ], 
        min_depth: int = 0
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Normalize wrapper-compatible and teacher-forcing inputs.

        Args:
            inputs (tuple[tf.Tensor, ...]): Three encoder tensors, optionally
                followed by an explicit decoder image.
            min_depth (int): Encoder resume depth. At depth zero the three-input
                form reuses the encoder image. At later depths it creates a
                zero decoder image because ``inputs[0]`` is a latent tensor.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Encoder input,
            timestep IDs, tensorized labels, and decoder image.

        Raises:
            ValueError: If ``inputs`` has neither three nor four tensors.
        """

        # Reuse encoder images as decoder inputs for the standard three-tensor call.
        if len(inputs) == 3:
            encoder_input, times, labels = inputs
            decoder_images = encoder_input if min_depth == 0 else tf.zeros(
                (
                    tf.shape(encoder_input)[0], 
                    self.decoder.current_resolution, 
                    self.decoder.current_resolution, 
                    self.decoder.channels, 
                ), 
                dtype=self.decoder.compute_dtype, 
            )
        # Accept an explicit decoder image for asymmetric encoder-decoder calls.
        elif len(inputs) == 4:
            encoder_input, times, labels, decoder_images = inputs
        # Reject input tuples outside the documented three- or four-tensor forms.
        else:
            raise ValueError(
                "inputs must contain images, times, labels, and optionally "
                "decoder_images."
            )

        return (
            encoder_input, 
            tf.convert_to_tensor(times), 
            tf.convert_to_tensor(labels), 
            decoder_images,
        )

    def _validate_decoder_features(
        self, 
        encoder_features: list[tf.Tensor | None], 
    ) -> None:
        """Reject decoder routes to unavailable encoder feature slots.

        Args:
            encoder_features (list[tf.Tensor | None]): Depth-indexed encoder
                outputs. Resumed or truncated encoding represents unavailable
                depths with ``None``.

        Returns:
            None: Valid feature routes are left unchanged.

        Raises:
            ValueError: If a configured decoder aggregator selects a skipped
                encoder depth, such as a pre-bottleneck VAE feature.
        """

        for mapping_name in (
            "feature_aggregation_ids_dict", 
            "cross_attention_aggregation_ids_dict", 
        ):
            for ids in getattr(self.decoder, mapping_name).values():
                # Ensure every decoder-routed encoder feature was actually computed.
                if any(
                    index >= len(encoder_features)
                    or encoder_features[index] is None
                    for index in ids
                ):
                    raise ValueError(
                        "decoder encoder-feature routes cannot select skipped "
                        "or unavailable encoder depths."
                    )

    def call(
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
        """Encode context and predict noise with the attached decoder.

        Args:
            inputs (tuple[tf.Tensor, ...]): ``(encoder_images, times, labels)``
                or ``(encoder_images, times, labels, decoder_images)``.  Images
                are float ``[B,H_e,W_e,C_e]`` and
                ``[B,H_d,W_d,C_d]`` respectively; times/labels are integer
                ``[B]``.  Three inputs reuse the encoder image in the decoder,
                so that tensor must satisfy both configured image interfaces.
            full_return (bool): Return the standard five-item transformer tuple
                when true.
            training (bool | None): Keras training mode for both submodels.
            min_depth (int): Encoder resume depth. ``0`` embeds an image;
                values ``1..depth`` treat ``inputs[0]`` as a matching encoder
                representation. Without a fourth input, the decoder starts
                from a zero image at its active resolution.

        Returns:
            tf.Tensor | tuple: Decoder image/noise ``[B,H,W,C]``. Full return is
            ``(noises, encoder_cond, encoder_features, encoder_regs,
            encoder_z_vals)``; condition, skipped features/regularizers, and
            absent latent statistics may contain ``None``.

        Raises:
            ValueError: If the input has neither three nor four tensors, or a
                decoder aggregator selects a skipped encoder feature.
        """

        encoder_images, times, labels, decoder_images = \
            self._split_encoder_decoder_inputs(inputs, min_depth)

        _, encoder_cond, encoder_features, encoder_regs, encoder_z = self.encode(
            (encoder_images, times, labels), 
            min_depth=min_depth, 
            training=training, 
        )
        self._validate_decoder_features(encoder_features)
        decoder_outputs = self.decoder(
            (decoder_images, times, labels), 
            encoder_cond, 
            encoder_features, 
            full_return=False, 
            training=training, 
        )
        noises = decoder_outputs["noises"]

        # Return encoder intermediates and auxiliary outputs only when requested.
        if full_return:
            return noises, encoder_cond, encoder_features, encoder_regs, encoder_z
        return noises

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
        """Run the standard encoder/decoder noise path.

        Args:
            inputs (tuple[tf.Tensor, ...]): Three- or four-input form accepted by
                :meth:`call`.
            full_return (bool): Include encoder condition, features,
                regularizers, and latent statistics.
            training (bool | None): Keras training mode.
            min_depth (int): Encoder resume depth forwarded to :meth:`call`.
                With three inputs, values above zero initialize the decoder
                image to zeros.

        Returns:
            tf.Tensor | tuple: Exactly the result contract of :meth:`call`.
        """

        return self.call(
            inputs, 
            full_return=full_return, 
            training=training, 
            min_depth=min_depth, 
        )

    def _apply_depths(
        self, 
        depth_spec: str | tuple | set | dict | list | None, 
    ) -> dict[str, dict[str, int]]:
        """Apply a validated encoder/decoder growth specification.

        An ordinary specification grows the encoder, matching
        :class:`DiffusionTransformer`. A targeted mapping accepts ``network``
        and ``decoder``; each value uses that branch's :meth:`add_depths`
        syntax. Encoder metadata is refreshed before decoder stages are built.

        Args:
            depth_spec (str | tuple | set | dict | list | None): Unscoped
                encoder specification or targeted branch mapping.

        Returns:
            dict[str, dict[str, int]]: Growth counts for ``network`` and
            ``decoder``.

        Raises:
            ValueError: If a targeted mapping contains another key.
        """

        targeted = isinstance(depth_spec, dict) and any(
            key in depth_spec for key in ("network", "decoder")
        )
        # Split targeted growth into encoder and decoder specifications.
        if targeted:
            # Reject targeted keys outside the two architecture branches.
            if not all(key in ("network", "decoder") for key in depth_spec):
                raise ValueError(
                    "targeted depth_spec keys must be 'network' or 'decoder'."
                )
            network_spec = depth_spec.get("network", [])
            decoder_spec = depth_spec.get("decoder", [])
        # Treat an unscoped specification as encoder-only growth.
        else:
            network_spec = depth_spec
            decoder_spec = []

        network_growth = DiffusionTransformer.add_depths(
            self, network_spec
        )["network"]
        dims, grids = self._get_encoder_feature_metadata()
        self.decoder.set_encoder_feature_metadata(dims, grids)
        decoder_growth = self.decoder.add_depths(decoder_spec)["network"]
        self._validate_decoder_output()
        self.decoder_kwargs = deepcopy(self.decoder.get_config())
        self._init_config["decoder_kwargs"] = deepcopy(self.decoder_kwargs)

        return {
            "network": network_growth,
            "decoder": decoder_growth,
        }

    def add_depths(
        self, 
        depth_spec: str | tuple | set | dict | list | None, 
    ) -> dict[str, dict[str, int]]:
        """Grow encoder and decoder branches transactionally.

        An ordinary specification grows the encoder, matching
        :class:`DiffusionTransformer`. A targeted mapping accepts ``network``
        and ``decoder`` values using the corresponding branch's normal
        progressive syntax. The complete change is first validated on an
        unbuilt configuration clone, so an invalid branch cannot leave the
        other branch partially grown.

        Args:
            depth_spec (str | tuple | set | dict | list | None): Unscoped
                encoder specification or targeted branch mapping.

        Returns:
            dict[str, dict[str, int]]: Before/added/after counts for
            ``network`` and ``decoder``.

        Raises:
            ValueError: If a target or layer name is unknown, or growth would
                violate an existing output-head contract.
        """

        probe_config = self.get_config()
        probe_config["build"] = False
        probe = DiTEncoderDecoder.from_config(probe_config)
        probe._apply_depths(deepcopy(depth_spec))

        return self._apply_depths(depth_spec)

    def add_class(self, source_network: object | None = None) -> None:
        """Grow the encoder and attached decoder label vocabularies together.

        Args:
            source_network (object | None): Optional already-expanded raw
                encoder-decoder used to initialize new EMA embedding rows.

        Returns:
            None: Both label vocabularies grow by one in place.
        """

        super().add_class(source_network=source_network)
        self.decoder.add_class(
            source_network=(
                source_network.decoder if source_network is not None else None
            )
        )

    def set_current_resolution(self, resolution: int | None = None) -> None:
        """Synchronize active encoder and decoder resolutions.

        Args:
            resolution (int | None): Positive size divisible by both patch
                sizes.  ``None`` restores each branch's configured image size.

        Returns:
            None: Both branches are updated in place.
        """

        encoder_resolution = self.image_size if resolution is None else resolution
        resolutions_and_patches = [(encoder_resolution, self.patch_size)]
        # Validate the decoder's derived resolution after combined updates.
        if hasattr(self, "decoder"):
            decoder_resolution = self.decoder.image_size if resolution is None \
                else resolution
            resolutions_and_patches.append(
                (decoder_resolution, self.decoder.patch_size)
            )
        for branch_resolution, patch_size in resolutions_and_patches:
            assert int(branch_resolution) == branch_resolution, \
                "resolution must be an integer."
            assert branch_resolution > 0, \
                "resolution must be positive."
            assert branch_resolution % patch_size == 0, \
                "resolution must be divisible by both patch sizes."

        DiffusionTransformer.set_current_resolution(self, resolution)
        # Propagate active resolution changes to the attached decoder.
        if hasattr(self, "decoder"):
            self.decoder.set_current_resolution(resolution)


def run_self_tests() -> dict[str, str]:
    """Test composition, generic-wrapper compatibility, and branch growth.

    Returns:
        dict[str, str]: One passed entry for :class:`DiTEncoderDecoder`.
    """

    import numpy as np
    from diffusion.models.wrapper.diffusion_model import DiffusionModel


    network_kwargs = {
        "num_classes": 2, 
        "timesteps": 8, 
        "image_size": 4, 
        "channels": 1, 
        "patch_size": 2, 
        "dim": 4, 
        "depth": 1, 
        "mha_num_heads": 1, 
        "vit_block_mlp_ratio": 1.0, 
        "build": False, 
    }
    decoder_kwargs = {
        "depth": 1, 
        "mha_num_heads": 1, 
        "vit_block_mlp_ratio": 1.0, 
        "shift_inputs": False, 
        "use_unpatchify": True, 
    }
    model = DiTEncoderDecoder(
        decoder_kwargs=decoder_kwargs, 
        name="encoder_decoder_test", 
        **network_kwargs, 
    )
    assert model.encoder is model
    assert model.use_unpatchify is False and model.decoder.use_unpatchify
    assert model.decoder.encoder_feature_dims == [4, 4]
    assert model.decoder.encoder_feature_grid_sizes == [2, 2]

    images = tf.reshape(tf.range(32, dtype=tf.float32), (2, 4, 4, 1)) / 32
    teacher = tf.reverse(images, axis=[1])
    times = tf.constant([1, 2], tf.int32)
    labels = tf.constant([1, 2], tf.uint8)
    three_input = model((images, times, labels), training=False)
    four_input = model((images, times, labels, images), training=False)
    assert three_input.shape == (2, 4, 4, 1)
    np.testing.assert_allclose(three_input, four_input, atol=1e-6)
    assert model((images, times, labels, teacher)).shape == (2, 4, 4, 1)
    full = model((images, times, labels), full_return=True, training=False)
    assert len(full) == 5
    assert full[0].shape == (2, 4, 4, 1)
    assert len(full[2]) == model.depth + 1
    predicted = model.predict_noise(
        (images, times, labels), full_return=True, training=False
    )
    np.testing.assert_allclose(predicted[0], full[0], atol=1e-6)

    _, cond, features, _, _ = DiffusionTransformer.encode(
        model, (images, times, labels), training=False
    )
    direct = model.decoder(
        (images, times, labels), 
        cond, features, 
        training=False
    )["noises"]
    np.testing.assert_allclose(direct, three_input, atol=1e-6)

    growth = model.add_depths({
        "network": "vision_transformer_block", 
        "decoder": "vision_transformer_block", 
    })
    assert growth == {
        "network": {"before": 1, "added": 1, "after": 2}, 
        "decoder": {"before": 1, "added": 1, "after": 2}, 
    }
    assert model.decoder.encoder_feature_dims == [4, 4, 4]
    assert model.decoder.layers_dicts[-1][model.decoder.VTB].__class__.__name__ \
        == "DiTDecoderBlock"
    assert model((images, times, labels)).shape == (2, 4, 4, 1)

    model.set_current_resolution(8)
    assert model.current_resolution == 8
    assert model.decoder.current_resolution == 8
    resized = tf.ones((1, 8, 8, 1))
    assert model((resized, times[:1], labels[:1])).shape == (1, 8, 8, 1)
    model.set_current_resolution()
    assert model.current_resolution == model.decoder.current_resolution == 4

    config = model.get_config()
    clone = DiTEncoderDecoder.from_config(config)
    assert clone.depth == 2 and clone.decoder.depth == 2
    assert clone.name == "encoder_decoder_test"
    assert clone((images, times, labels)).shape == (2, 4, 4, 1)

    wrapped_network = DiTEncoderDecoder(
        decoder_kwargs=decoder_kwargs, 
        **network_kwargs
    )
    wrapper = DiffusionModel(
        network=wrapped_network, 
        use_ema=True, 
        test_steps=2, 
    )
    wrapper.compile(optimizer="adam", loss="mse", run_eagerly=True)
    train_result = wrapper.train_step((images, tf.constant([0, 1], tf.uint8)))
    assert "noise_loss" in train_result and tf.math.is_finite(
        train_result["noise_loss"]
    )
    call_result = wrapper.call_network(
        images, times, labels, 
        network_name="raw", 
        training=False
    )
    assert call_result[0][0].shape == (2, 4, 4, 1)
    assert len(wrapper.network.weights) == len(wrapper.ema_network.weights)

    vae_network = DiTEncoderDecoder(
        decoder_kwargs=decoder_kwargs, 
        num_classes=2, 
        timesteps=8, 
        image_size=4, 
        channels=1, 
        patch_size=2, 
        dim=4, 
        depth=2, 
        mha_num_heads=1, 
        vit_block_mlp_ratio=1.0, 
        vit_block_ids=[], 
        reshaper_ids_dict={1: "flatten", 2: "unflatten"}, 
        reshaper_kwargs={"add_kl": True, "latent_dim_ratio": 0.5}, 
        build=False, 
    )
    assert vae_network.decoder.encoder_feature_grid_sizes == [2, None, 2]
    vae_wrapper = DiffusionModel(
        network=vae_network, 
        use_ema=False, 
        swap_noise_image=True, 
        test_steps=2, 
    )
    vae_sample = vae_wrapper.sample_vae(
        network_name="raw", labels=[1, 2]
    )
    assert vae_sample.shape == (2, 4, 4, 1)
    assert bool(tf.reduce_all(tf.math.is_finite(vae_sample)))

    try:
        model.add_depths({"encoder": "vision_transformer_block"})
    except ValueError:
        pass
    else:
        raise AssertionError("The public targeted key is named network.")
    before_depths = (model.depth, model.decoder.depth)
    try:
        model.add_depths({
            "network": "vision_transformer_block",
            "decoder": "not_a_layer",
        })
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid branch growth must fail preflight.")
    assert (model.depth, model.decoder.depth) == before_depths
    try:
        model((images, times))
    except ValueError:
        pass
    else:
        raise AssertionError("Only three or four composite inputs are valid.")
    resumed = model(
        (full[2][-1], times, labels), 
        min_depth=1, 
        training=False
    )
    assert resumed.shape == (2, 4, 4, 1)
    try:
        DiTEncoderDecoder(
            decoder_kwargs={
                **decoder_kwargs, 
                "encoder_feature_dims": [4], 
            }, 
            **network_kwargs, 
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Decoder metadata must match the encoder.")
    try:
        DiTEncoderDecoder(
            decoder_kwargs={
                **decoder_kwargs, 
                "cls_token_type": "new_weight", 
                "feature_aggregation_ids_dict": {1: [0]}, 
            }, 
            **network_kwargs, 
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Routed encoder/decoder class-token settings must match."
        )
    try:
        DiTEncoderDecoder(
            decoder_kwargs={
                **decoder_kwargs, 
                "cls_token_type": "new_weight", 
                "cross_attention_aggregation_ids_dict": {1: [0]}, 
                "cross_attention_plug_type": "queries", 
            }, 
            **network_kwargs,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Cross-attention queries must match decoder token counts."
        )
    unequal_value_tokens = DiTEncoderDecoder(
        decoder_kwargs={
            **decoder_kwargs, 
            "cls_token_type": "new_weight", 
            "cross_attention_aggregation_ids_dict": {1: [0]}, 
            "cross_attention_plug_type": "values", 
        }, 
        **network_kwargs, 
    )
    assert unequal_value_tokens(
        (images, times, labels), training=False
    ).shape == (2, 4, 4, 1)
    try:
        DiTEncoderDecoder(
            decoder_kwargs={**decoder_kwargs, "cond_dim": 8}, 
            **network_kwargs, 
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Shared decoder conditioning must match cond_dim.")
    separate_cond = DiTEncoderDecoder(
        decoder_kwargs={
            **decoder_kwargs, 
            "cond_dim": 8, 
            "decoder_separate_cond": True, 
        }, 
        **network_kwargs, 
    )
    assert separate_cond((images, times, labels)).shape == (2, 4, 4, 1)
    for undersized_decoder in (
        {"timesteps": 4}, 
        {"num_classes": 1}, 
    ):
        try:
            DiTEncoderDecoder(
                decoder_kwargs={
                    **decoder_kwargs, 
                    **undersized_decoder, 
                    "decoder_separate_cond": True, 
                }, 
                **network_kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Decoder embedding tables must cover wrapper-visible IDs."
            )

    tf.keras.backend.clear_session()
    return {"DiTEncoderDecoder": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
