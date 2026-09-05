"""Decoder-style diffusion transformer with explicit encoder-context routing.

DiTDecoder owns token embedding, optional causal attention, routed encoder
features, and image reconstruction. It accepts packed Keras context inputs or
separate encoder condition/features. Diffusion schedules, losses, EMA, and
sampling belong to the wrapper; composite encoder-decoder classes construct
and validate the encoder metadata used here.
"""

import tensorflow as tf
from tensorflow.keras import layers

from copy import deepcopy

from common.validation import require

from diffusion.layers.block.di_t_decoder_block import DiTDecoderBlock
from diffusion.layers.feature_handler import FeatureHandler
from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer


class DiTDecoder(DiffusionTransformer):
    """Decode image tokens while attending to encoder-side representations.

    Decoder depth 0 is the embedded, optionally right-shifted input.  Depths
    1..N use the same connector, transformer, mixer, scaler, reshaper, and
    regularizer pattern as :class:`DiffusionTransformer`, with two additional
    handlers that select encoder features.  A feature aggregator replaces or
    augments the decoder stream; a cross-attention aggregator supplies external
    queries or values.  Without an explicit cross-attention handler, decoder
    blocks attend to the final available encoder feature.

    Calls accept either the legacy three-part decoder input plus separate
    ``encoder_cond`` and ``encoder_features_list`` arguments, or one flat tuple
    ``(images, times, labels, encoder_cond, *encoder_features)``.  The latter is
    the symbolic/Keras interface.  :meth:`encode`, :meth:`predict_noise`, and
    :meth:`add_depths` follow the corresponding transformer APIs while retaining
    the required encoder context.

    Attributes:
        encoder_output_grid_size (int): Final encoder spatial grid side supplied
            at construction; used to infer omitted one-feature metadata.
        encoder_output_dim (int): Final encoder feature width.
        encoder_feature_dims (list[int]): Widths indexed by encoder depth, copied
            from supplied metadata or initialized from encoder_output_dim.
        encoder_feature_grid_sizes (list[int | None]): Matching spatial sides;
            None represents a feature without a square spatial grid.
        encoder_feature_is_flat (list[bool]): Explicit rank-two state per encoder
            feature. False with a missing grid represents non-square token data.
        decoder_separate_cond (bool): Whether decoder-owned condition embeddings
            replace the supplied encoder condition during decoding.
        use_causal_mask (bool): Whether decoder self-attention is lower triangular.
        feature_aggregation_ids_dict (dict[int, list[int]]): Normalized routes
            injecting encoder features into the decoder's main stream.
        cross_attention_aggregation_ids_dict (dict[int, list[int]]): Normalized
            routes supplying encoder features to cross attention.
        layers_dicts (list[dict[str, tf.keras.layers.Layer]]): Inherited ordered
            decoder stages, including optional encoder feature handlers. Other
            inherited state and constructor controls follow DiffusionTransformer.
    """

    FA  = "0_feature_aggregator"
    FC  = "1_feature_connector"
    CAA = "2_cross_attention_aggregator"
    CAC = "3_cross_attention_connector"
    VTB = "4_vision_transformer_block"
    LM  = "5_local_mixer"
    DS  = "6_downsampler"
    US  = "7_upsampler"
    R   = "8_reshaper"
    CTR = "9_cls_token_regularizer"

    def __init__(
        self, 
        encoder_output_grid_size: int, 
        encoder_output_dim: int, 
        encoder_feature_grid_sizes: list[int | None] | tuple[int | None, ...] | None = None, 
        encoder_feature_dims: list[int] | tuple[int, ...] | None = None, 
        shift_inputs: bool = True, 
        use_decoder_ids: list[int | None] = [None], 
        decoder_separate_cond: bool = False, 
        use_causal_mask: bool = True, 
        feature_aggregation_ids_dict: dict[
            int, list[int | None] | tuple[int | None, ...]
        ] = {}, 
        feature_aggregation_kwargs: dict = {}, 
        cross_attention_aggregation_ids_dict: dict[
            int, list[int | None] | tuple[int | None, ...]
        ] = {}, 
        cross_attention_aggregation_kwargs: dict = {}, 
        build: bool = True, 
        encoder_feature_is_flat: list[bool] | tuple[bool, ...] | None = None,
        **kwargs: object
    ) -> None:
        """Initialize decoder configuration and optionally build the model.

        Args:
            encoder_output_grid_size (int): Final encoder token-grid side.
            encoder_output_dim (int): Final encoder feature width.
            encoder_feature_grid_sizes (list[int | None] | None): Grid side at every encoder feature
                depth. ``None`` creates one entry from ``encoder_output_grid_size``. A ``None`` item
                denotes a flat non-spatial feature. Defaults to ``None``.
            encoder_feature_dims (list[int] | None): Feature width at every encoder depth. ``None``
                creates one final-feature entry from ``encoder_output_dim``. The list index is the ID
                used by the two encoder aggregation dictionaries. Defaults to ``None``.
            encoder_feature_is_flat (list[bool] | None): Explicit rank state for every encoder feature.
                ``None`` preserves the legacy rule that a missing grid denotes a rank-2 feature;
                explicit ``False`` distinguishes non-square rank-3 token sequences. Defaults to
                ``None``.
            shift_inputs (bool): Right-shift decoder patch tokens by prepending the shared learned BOS
                token; true by default for autoregressive teacher forcing. Defaults to ``True``.
            use_decoder_ids (list[int | None]): Depths implemented with causal- capable
                ``DiTDecoderBlock``. ``[None]`` expands to every decoder depth; ``[]`` selects
                encoder-style blocks. Defaults to ``[None]``.
            decoder_separate_cond (bool): Build the decoder condition from its own time/label embedders
                when true. False uses ``encoder_cond`` while retaining only embedders required by an
                optional condition-derived decoder class token. Defaults to ``False``.
            use_causal_mask (bool): Supply a lower-triangular attention mask to decoder blocks. Defaults
                to ``True``.
            feature_aggregation_ids_dict (dict[int, list[int | None]]): Maps a decoder target depth in
                ``1..depth`` to encoder feature IDs. ``-1`` is the final encoder feature and ``None``
                selects all encoder depths. Example: ``{2: (0, -1)}`` merges the first and final encoder
                features before decoder depth 2. Defaults to ``{}``.
            feature_aggregation_kwargs (dict[str, object]): Shared ``FeatureHandler`` options:
                ``connect_axis`` (int), ``connect_type`` (``"concat"``/``"add"``), ``use_layer_norm``
                (bool), ``ln_dim`` (int | None), ``ln_mlp_ratio`` (float | None), ``ln_no_adaptation``
                (bool), ``mlp_output_dim`` (int | None), ``mlp_ratio`` (float | None), and
                ``mlp_activation_func`` (Keras activation). Unknown keys raise ``AssertionError``.
                Rank-3 token features use axis ``1``/``-2`` for tokens and ``2``/``-1`` for channels;
                flattened rank-2 features accept only ``1``/``-1``. Defaults to ``{}``.
            cross_attention_aggregation_ids_dict (dict[int, list[int | None]]): Maps decoder depths to
                encoder features used as cross- attention values or queries. It uses the same depth and
                ID syntax as ``feature_aggregation_ids_dict``. Defaults to ``{}``.
            cross_attention_aggregation_kwargs (dict[str, object]): Encoder cross-attention handler
                options with the same accepted keys as ``feature_aggregation_kwargs``. Defaults to
                ``{}``.
            build (bool): Build the packed symbolic interface immediately. It contains four fixed inputs
                plus one input per encoder feature metadata entry. Set false to defer variable creation.
                Defaults to ``True``.
            **kwargs (object): ``DiffusionTransformer`` arguments (for example ``depth``,
                connection ID mappings, block IDs, dimensions, and output-head
                options) plus standard Keras ``Model`` keys ``name``,
                ``trainable``, ``dtype``, and ``dynamic``.

        Returns:
            None: Decoder layers and configuration are initialized in place.

        Raises:
            AssertionError: If constructor dimensions, routes, options, or compatible
                conditioning/token combinations violate the network contract.
            ValueError: If a nested layer cannot construct the requested shape or
                mode, including an unusable variational latent width.
        """

        encoder_output_grid_size = int(encoder_output_grid_size)
        encoder_output_dim = int(encoder_output_dim)
        raw_feature_ids = deepcopy(feature_aggregation_ids_dict)
        raw_cross_ids = deepcopy(cross_attention_aggregation_ids_dict)
        # Copy explicit encoder feature widths or derive one final-feature width.
        feature_dims = [
            int(dim) for dim in encoder_feature_dims
        ] if encoder_feature_dims is not None else [encoder_output_dim]
        # Copy explicit encoder grids or derive one final-feature grid.
        # Preserve nonspatial metadata as None while normalizing numeric grid sides.
        feature_grids = [
            None if grid is None else int(grid)
            for grid in encoder_feature_grid_sizes
        ] if encoder_feature_grid_sizes is not None else [
            encoder_output_grid_size
        ]
        # Use explicit rank metadata or infer flattened features from missing grids.
        feature_is_flat = list(encoder_feature_is_flat) if \
            encoder_feature_is_flat is not None else [
                grid is None for grid in feature_grids
            ]

        # The base constructor dynamically calls the decoder factory, so these
        # values must exist before ``DiffusionTransformer.__init__`` creates
        # ``layers_dicts``.
        object.__setattr__(self, "encoder_output_grid_size", encoder_output_grid_size)
        object.__setattr__(self, "encoder_output_dim", encoder_output_dim)
        object.__setattr__(self, "encoder_feature_grid_sizes", feature_grids)
        object.__setattr__(self, "encoder_feature_dims", feature_dims)
        object.__setattr__(self, "encoder_feature_is_flat", feature_is_flat)
        object.__setattr__(self, "decoder_separate_cond", decoder_separate_cond)
        object.__setattr__(self, "use_causal_mask", use_causal_mask)
        object.__setattr__(self, "feature_aggregation_ids_dict", raw_feature_ids)
        object.__setattr__(self, "feature_aggregation_kwargs", deepcopy(feature_aggregation_kwargs))
        object.__setattr__(self, "cross_attention_aggregation_ids_dict", raw_cross_ids)
        object.__setattr__(self, "cross_attention_aggregation_kwargs", deepcopy(cross_attention_aggregation_kwargs))

        super().__init__(
            shift_inputs=shift_inputs, 
            use_decoder_ids=use_decoder_ids, 
            build=False, 
            **kwargs
        )

        self.build_ = build
        self._init_config.update({
            "encoder_output_grid_size": encoder_output_grid_size, 
            "encoder_output_dim": encoder_output_dim, 
            "encoder_feature_grid_sizes": deepcopy(feature_grids), 
            "encoder_feature_dims": deepcopy(feature_dims), 
            "encoder_feature_is_flat": deepcopy(feature_is_flat),
            "decoder_separate_cond": decoder_separate_cond, 
            "use_causal_mask": use_causal_mask, 
            "feature_aggregation_ids_dict": raw_feature_ids, 
            "feature_aggregation_kwargs": deepcopy(feature_aggregation_kwargs), 
            "cross_attention_aggregation_ids_dict": raw_cross_ids, 
            "cross_attention_aggregation_kwargs": deepcopy(
                cross_attention_aggregation_kwargs
            ), 
            "build": build, 
        })

        # Retain only embedders needed by shared class/distillation tokens.
        if not self.decoder_separate_cond:
            # Retain a decoder time lookup only when a decoder token still consumes time.
            self.time_embedder = self.time_embedder \
                if "time" in self._cls_token_type or \
                "time" in self._distil_token_type else None
            # Retain a decoder label lookup only for token conditions or depth-zero
            # regularization.
            self.label_embedder = self.label_embedder \
                if "label" in self._cls_token_type or \
                "label" in self._distil_token_type or \
                0 in self.cls_token_regularizer_ids else None
            # Retain a condition merger only when a decoder token combines time and labels.
            self.conds_merger = self.conds_merger \
                if ("time" in self._cls_token_type and \
                "label" in self._cls_token_type) or \
                ("time" in self._distil_token_type and \
                "label" in self._distil_token_type) else None
        # Restore label embeddings needed by a depth-zero label regularizer.
        if 0 in self.cls_token_regularizer_ids and self.label_embedder is None:
            self.label_embedder = self._create_label_embedder()

        # Materialize decoder variables when eager construction is requested.
        if self.build_:
            self.build()

    def _check_assertions(self, local_vars: dict) -> None:
        """Validate base and decoder-specific constructor arguments.

        Args:
            local_vars (dict[str, object]): Base transformer constructor values.

        Returns:
            None: Invalid metadata, IDs, or handler options raise
            ``AssertionError``.

        Raises:
            AssertionError: If shape, conditioning, depth routes, or component-option
                constraints required by this constructor are violated.
        """

        base_local_vars = local_vars
        regularizer_ids = local_vars.get("cls_token_regularizer_ids", [])
        # Collapse an explicit all-depth regularizer sequence back to its None sentinel.
        if isinstance(regularizer_ids, (list, tuple, set, frozenset)) and \
        len(regularizer_ids) == local_vars["depth"] + 1 and \
        set(regularizer_ids) == set(range(local_vars["depth"] + 1)):
            base_local_vars = dict(local_vars)
            base_local_vars["cls_token_regularizer_ids"] = [None]
        super()._check_assertions(base_local_vars)
        require(len(self.encoder_feature_dims) > 0)
        require(
            self.encoder_output_grid_size > 0,
            "encoder_output_grid_size must be positive."
        )
        require(
            self.encoder_output_dim > 0,
            "encoder_output_dim must be positive."
        )
        require(
            all(dim > 0 for dim in self.encoder_feature_dims),
            "encoder feature dimensions must be positive."
        )
        require(
            all(
                grid is None or grid > 0
                for grid in self.encoder_feature_grid_sizes
            ),
            "encoder feature grid sizes must be positive when supplied."
        )
        require(len(self.encoder_feature_dims) == len(
            self.encoder_feature_grid_sizes
        ), "encoder feature dimensions and grid sizes must have equal length.")
        require(len(self.encoder_feature_dims) == len(
            self.encoder_feature_is_flat
        ), "encoder feature dimensions and rank states must have equal length.")
        require(all(
            not is_flat or grid is None
            for grid, is_flat in zip(
                self.encoder_feature_grid_sizes,
                self.encoder_feature_is_flat,
            )
        ), "flat encoder features cannot have a spatial grid.")
        require(self.encoder_feature_dims[-1] == self.encoder_output_dim)
        require(self.encoder_feature_grid_sizes[-1] == self.encoder_output_grid_size)
        require(not self.encoder_feature_is_flat[-1])

        for mapping_name in (
            "feature_aggregation_ids_dict", 
            "cross_attention_aggregation_ids_dict", 
        ):
            mapping = getattr(self, mapping_name)
            require(all(
                1 <= key <= local_vars["depth"]
                for key in mapping
            ), f"keys in {mapping_name} must be decoder depths in [1, depth].")

        for kwargs_name in (
            "feature_aggregation_kwargs", 
            "cross_attention_aggregation_kwargs", 
            "connection_kwargs", 
            "cross_attention_kwargs", 
        ):
            handler_kwargs = getattr(
                self, kwargs_name, local_vars.get(kwargs_name, {})
            )
            invalid = set(handler_kwargs) - set(
                self.feature_handler_kwargs_allowed_vals
            )
            require(not invalid, f"Unknown keys in {kwargs_name}: {sorted(invalid)}.")
            require(handler_kwargs.get("connect_axis", -1) in (
                -2, -1, 1, 2
            ), "decoder feature handlers support token or channel axes only.")

    def _normalize_encoder_ids(
        self, 
        ids: int | list[int | None] | tuple[int | None, ...] | None
    ) -> list[int]:
        """Normalize encoder feature IDs against the metadata list.

        Args:
            ids (int | list[int | None] | tuple[int | None, ...] | None): One
                or more encoder IDs. ``None`` selects every feature.

        Returns:
            list[int]: Absolute IDs in input order.
        """

        count = len(self.encoder_feature_dims)
        # Normalize a single encoder feature ID to a list.
        values = [ids] if isinstance(ids, int) else list(ids or [None])
        # Expand the None sentinel to all available encoder feature IDs.
        if None in values:
            return list(range(count))

        # Resolve negative encoder IDs relative to the encoder feature count.
        normalized = [value + count if value < 0 else value for value in values]
        require(all(0 <= value < count for value in normalized), (
            "encoder feature IDs must reference encoder_feature_dims."
        ))

        return normalized

    def _handle_all_ids(self) -> None:
        """Normalize base decoder IDs and encoder aggregation IDs.

        Returns:
            None: Runtime mappings contain absolute integer IDs.
        """

        super()._handle_all_ids()
        for mapping_name in (
            "feature_aggregation_ids_dict",
            "cross_attention_aggregation_ids_dict",
        ):
            setattr(self, mapping_name, {
                key: self._normalize_encoder_ids(ids)
                for key, ids in getattr(self, mapping_name).items()
            })

    def _create_encoder_feature_handler(
        self, 
        ids: list[int], 
        increased_dim: int = 0, 
        second_grid_size: int | None = None, 
        second_is_flat: bool = False,
        output_dim_flag: bool = True, 
        kwargs: dict | None = None, 
        name: str | None = None
    ) -> FeatureHandler:
        """Create a handler whose source dimensions come from the encoder.

        Args:
            ids (list[int]): Encoder feature IDs.
            increased_dim (int): Width of an appended decoder-side tensor. Defaults to ``0``.
            second_grid_size (int | None): Grid of that decoder tensor. Defaults to ``None``.
            second_is_flat (bool): Whether that decoder tensor is rank 2. Defaults to ``False``.
            output_dim_flag (bool): Allow automatic projection back to ``self.dim``. Defaults to
                ``True``.
            kwargs (dict | None): Valid ``FeatureHandler`` overrides. Defaults to ``None``.
            name (str | None): Generated Keras layer name. Defaults to ``None``.

        Returns:
            FeatureHandler: Configured encoder feature selector/merger.
        """

        # Use default encoder-aggregation options when no mapping is supplied.
        kwargs = {} if kwargs is None else kwargs
        dims = [self.encoder_feature_dims[index] for index in ids]
        grids = [self.encoder_feature_grid_sizes[index] for index in ids]
        flat_states = [self.encoder_feature_is_flat[index] for index in ids]
        # Include an appended decoder stream when calculating merged source metadata.
        if increased_dim:
            dims.append(increased_dim)
            grids.append(second_grid_size)
            flat_states.append(second_is_flat)
        # Concatenation sums source widths; addition preserves the first source width.
        merged_dim = sum(dims) if kwargs.get("connect_type", "concat") == "concat" else dims[0]
        grid_size = grids[0]
        output_is_flat = flat_states[0]
        # Project widened encoder aggregates back to the forced decoder width when allowed.
        options = {
            "ids": ids, 
            "ln_dim": merged_dim, 
            "mlp_output_dim": self.dim if self.dim_forced and output_dim_flag
                and merged_dim > self.dim else None, 
            "ln_mlp_ratio": self.ln_mlp_ratio, 
            "ln_no_adaptation": self.ln_no_adaptation, 
            "grid_size": grid_size,
            "name": name, 
        }
        options.update(kwargs)
        handler = FeatureHandler(**options)
        handler.output_grid_size = handler.grid_size
        handler.output_is_flat = output_is_flat

        return handler

    def _create_feature_handler(
        self, 
        ids_set: list[int], 
        layers_dicts: list[dict], 
        base_dim: int, 
        base_grid_size: int | None,
        dim_forced: bool, 
        ln_mlp_ratio: float, 
        ln_no_adaptation: bool, 
        kwargs: dict, 
        zero_index_base_dim: int | None = None, 
        base_is_flat: bool = False,
        increased_dim: int = 0,
        increased_grid_size: int | None = None,
        increased_is_flat: bool = False,
        output_dim_flag: bool = True, 
        prepended_tokens_num: int | None = None,
        name: str | None = None, 
    ) -> FeatureHandler:
        """Create an internal connector with accurate width/grid metadata.

        Args:
            ids_set (list[int]): Decoder feature depths to select.
            layers_dicts (list[dict]): Previously created decoder stages.
            base_dim (int): Forced output width and depth-zero width.
            base_grid_size (int | None): Decoder depth-zero grid metadata.
            dim_forced (bool): Project widened channel concatenation back to
                ``base_dim``.
            ln_mlp_ratio (float | None): Adaptive-normalization MLP ratio.
            ln_no_adaptation (bool): Disable condition adaptation when true.
            kwargs (dict[str, object]): ``FeatureHandler`` options.
            zero_index_base_dim (int | None): Optional depth-zero width. Defaults to ``None``.
            base_is_flat (bool): Compatibility metadata currently unused by this forwarding method;
                source geometry is inferred by the base connector factory. Defaults to ``False``.
            increased_dim (int): Width of an appended secondary tensor. Defaults to ``0``.
            increased_grid_size (int | None): Compatibility metadata currently unused by this forwarding
                method; the appended feature width is forwarded separately through increased_dim.
                Defaults to ``None``.
            increased_is_flat (bool): Compatibility rank flag currently unused by this forwarding
                method. Defaults to ``False``.
            output_dim_flag (bool): Permit automatic forced projection. Defaults to ``True``.
            prepended_tokens_num (int | None): Compatibility prefix-count override currently unused by
                this forwarding method; the inherited connector factory uses the main network prefix
                count. Defaults to ``None``.
            name (str | None): Keras layer name. Defaults to ``None``.

        Returns:
            FeatureHandler: Inherited selector/merger configured from source feature
            widths and the common source grid; compatibility metadata is not added here.
        """

        return super()._create_feature_handler(
            ids_set=ids_set,
            layers_dicts=layers_dicts,
            base_dim=base_dim,
            base_grid_size=base_grid_size,
            dim_forced=dim_forced,
            ln_mlp_ratio=ln_mlp_ratio,
            ln_no_adaptation=ln_no_adaptation,
            kwargs=kwargs,
            zero_index_base_dim=zero_index_base_dim,
            increased_dim=increased_dim,
            output_dim_flag=output_dim_flag,
            name=name,
        )

    def _get_layers_dict_last_output_dim(
        self, 
        layers_dict: dict, 
        skip_reshaper: bool
    ) -> int | None:
        """Return the last decoder-stage width, including aggregation.

        Args:
            layers_dict (dict): One decoder stage.
            skip_reshaper (bool): Ignore reshape output widths.

        Returns:
            int | None: Last known feature width.
        """

        output_dim = super()._get_layers_dict_last_output_dim(
            layers_dict, skip_reshaper
        )
        # Use the aggregator width when no later component changes it.
        if output_dim is None and self.FA in layers_dict:
            return layers_dict[self.FA].output_dim
        return output_dim

    def _get_last_grid_size(
        self, 
        i: int, 
        layers_dicts: list[dict], 
        base_grid_size: int | None,
        skip_reshaper: bool = False,
        base_is_flat: bool = False,
    ) -> int | None:
        """Resolve decoder grid size, including encoder aggregation.

        Args:
            i (int): Zero-based decoder stage, or ``-1`` for depth 0.
            layers_dicts (list[dict]): Decoder stage dictionaries.
            base_grid_size (int): Depth-zero grid side forwarded to the inherited resolver, or None when
                no spatial side is available.
            skip_reshaper (bool): Ignore reshaper output rank changes. Defaults to ``False``.

            base_is_flat (bool): Compatibility argument currently unused by this forwarding method. Flat
                state is inferred by the inherited resolver from reshape components and the zero-grid
                sentinel. Defaults to ``False``.

        Returns:
            int | None: Latest square spatial side. A flattened rank-two reshaper
            output uses integer sentinel 0; None can propagate when the supplied
            base grid itself is unknown. skip_reshaper=True ignores rank changes.
        """

        return super()._get_last_grid_size(
            i,
            layers_dicts,
            base_grid_size,
            skip_reshaper=skip_reshaper,
        )

    def _create_layers_dict(self, i: int, layers_dicts: list[dict]) -> dict:
        """Create one complete decoder stage in execution order.

        Args:
            i (int): Zero-based stage index.
            layers_dicts (list[dict]): Previously created decoder stages.

        Returns:
            dict: Selected aggregator, connector, attention, spatial,
            reshaping, and regularizer layers.
        """

        stage = {}
        key = i + 1
        previous_dim = self._get_last_output_dim(i - 1, layers_dicts, self.dim)
        previous_grid = self._get_last_grid_size(
            i - 1, layers_dicts, self.grid_size
        )
        previous_is_flat = previous_grid == 0

        # Build the encoder aggregator and append decoder features unless separately connected.
        if key in self.feature_aggregation_ids_dict:
            append_current = key not in self.connection_ids_dict
            # Include current decoder width only when its stream joins the encoder aggregate.
            # Include the appended decoder grid only when its stream is being aggregated.
            # Carry the appended stream's flat/spatial state only when it is included.
            stage[self.FA] = self._create_encoder_feature_handler(
                self.feature_aggregation_ids_dict[key],
                increased_dim=previous_dim if append_current else 0,
                second_grid_size=previous_grid if append_current else None,
                second_is_flat=previous_is_flat if append_current else False,
                output_dim_flag=key not in self.connection_ids_dict,
                kwargs=self.feature_aggregation_kwargs,
                name=f"{self.name_prefix}depth_{key}_{self.FA[2:]}",
            )

        # Build this depth's decoder residual feature connector.
        if key in self.connection_ids_dict:
            # Include encoder-aggregate width in a decoder connector only when that aggregate
            # exists.
            # Forward encoder-aggregate grid metadata only when the stage has an aggregator.
            # Forward encoder-aggregate rank metadata only when the stage has an aggregator.
            stage[self.FC] = self._create_feature_handler(
                ids_set=self.connection_ids_dict[key],
                layers_dicts=layers_dicts,
                base_dim=self.dim,
                base_grid_size=self.grid_size,
                dim_forced=self.dim_forced,
                ln_mlp_ratio=self.ln_mlp_ratio,
                ln_no_adaptation=self.ln_no_adaptation,
                increased_dim=stage[self.FA].output_dim
                    if self.FA in stage else 0,
                increased_grid_size=stage[self.FA].output_grid_size
                    if self.FA in stage else None,
                increased_is_flat=stage[self.FA].output_is_flat
                    if self.FA in stage else False,
                kwargs=self.connection_kwargs,
                name=f"{self.name_prefix}depth_{key}_{self.FC[2:]}",
            )

        # Build the encoder cross-attention feature aggregator.
        if key in self.cross_attention_aggregation_ids_dict:
            stage[self.CAA] = self._create_encoder_feature_handler(
                self.cross_attention_aggregation_ids_dict[key],
                output_dim_flag=key not in self.cross_attention_ids_dict,
                kwargs=self.cross_attention_aggregation_kwargs,
                name=f"{self.name_prefix}depth_{key}_{self.CAA[2:]}",
            )

        # Build this depth's decoder cross-attention connector.
        if key in self.cross_attention_ids_dict:
            # Include encoder cross-aggregate width only when that attention source exists.
            # Carry encoder cross-aggregate grid metadata only when available.
            # Carry encoder cross-aggregate rank metadata only when available.
            stage[self.CAC] = self._create_feature_handler(
                ids_set=self.cross_attention_ids_dict[key],
                layers_dicts=layers_dicts,
                base_dim=self.dim,
                base_grid_size=self.grid_size,
                dim_forced=self.dim_forced,
                ln_mlp_ratio=self.ln_mlp_ratio,
                ln_no_adaptation=self.ln_no_adaptation,
                increased_dim=stage[self.CAA].output_dim
                    if self.CAA in stage else 0,
                increased_grid_size=stage[self.CAA].output_grid_size
                    if self.CAA in stage else None,
                increased_is_flat=stage[self.CAA].output_is_flat
                    if self.CAA in stage else False,
                kwargs=self.cross_attention_kwargs,
                name=f"{self.name_prefix}depth_{key}_{self.CAC[2:]}",
            )

        # Build a transformer block using the widths produced by feature handlers.
        if key in self.vit_block_ids:
            query_dim = None
            # Derive query shape from the configured query-side feature handler.
            if self.cross_attention_plug_type == "queries":
                # Use connected decoder features as attention queries when available.
                if self.CAC in stage:
                    query_dim = stage[self.CAC].output_dim
                # Otherwise use aggregated encoder features as attention queries.
                elif self.CAA in stage:
                    query_dim = stage[self.CAA].output_dim
                # Prefer self-connected query grids, then encoder aggregates, then the previous
                # grid.
                # Use an encoder query aggregate when no classifier self-connector supplies
                # queries.
                query_grid = stage[self.CAC].grid_size \
                    if self.CAC in stage else stage[self.CAA].output_grid_size \
                    if self.CAA in stage else previous_grid
                # Prefer the decoder connector grid, then encoder aggregation, then the prior
                # stream.
                # Use the encoder-aggregate grid when no decoder connector replaces it.
                decoder_grid = stage[self.FC].grid_size \
                    if self.FC in stage else stage[self.FA].output_grid_size \
                    if self.FA in stage else previous_grid
                require(query_grid == decoder_grid, (
                    "cross-attention queries must match the decoder token grid."
                ))

            stage[self.VTB] = self._create_vit_block(
                i=i,
                layers_dicts=layers_dicts,
                layers_dict=stage,
                base_dim=self.dim,
                base_grid_size=self.grid_size,
                mha_key_dim=self.mha_key_dim,
                mha_value_dim=self.mha_value_dim,
                mha_query_dim=query_dim,
                mha_num_heads=self.mha_num_heads,
                mlp_ratio=self.vit_block_mlp_ratio,
                mlp_output_dim=self.vit_block_mlp_output_dims.get(key),
                ln_mlp_ratio=self.ln_mlp_ratio,
                ln_no_adaptation=self.ln_no_adaptation,
                drop_prob=self.drop_prob,
                drop_per_sample=self.drop_per_sample,
                use_decoder=key in self.use_decoder_ids,
                name_prefix=f"{self.name_prefix}depth_{key}_",
            )

        # Build this decoder depth's local mixer.
        if key in self.local_mixer_ids:
            stage[self.LM] = self._create_local_mixer(
                i=i,
                dim_forced=self.dim_forced,
                layers_dicts=layers_dicts,
                layers_dict=stage,
                base_dim=self.dim,
                base_grid_size=self.grid_size,
                ln_mlp_ratio=self.ln_mlp_ratio,
                ln_no_adaptation=self.ln_no_adaptation,
                circumvent_tokens=int(self.cls_token_type is not None) +
                    int(self.distil_token_type is not None),
                kwargs=self.local_mixer_kwargs,
                name=f"{self.name_prefix}depth_{key}_{self.LM[2:]}",
            )

        # Build this decoder depth's downsampler.
        if key in self.downsample_ids:
            stage[self.DS] = self._create_scaler(
                scaler_type="downsample",
                i=i,
                dim_forced=self.dim_forced,
                layers_dicts=layers_dicts,
                layers_dict=stage,
                base_dim=self.dim,
                base_grid_size=self.grid_size,
                ln_mlp_ratio=self.ln_mlp_ratio,
                ln_no_adaptation=self.ln_no_adaptation,
                circumvent_tokens=int(self.cls_token_type is not None) +
                    int(self.distil_token_type is not None),
                kwargs=self.downsample_kwargs,
                name=f"{self.name_prefix}depth_{key}_{self.DS[2:]}",
            )

        # Build this decoder depth's upsampler.
        if key in self.upsample_ids:
            stage[self.US] = self._create_scaler(
                scaler_type="upsample",
                i=i,
                dim_forced=self.dim_forced,
                layers_dicts=layers_dicts,
                layers_dict=stage,
                base_dim=self.dim,
                base_grid_size=self.grid_size,
                ln_mlp_ratio=self.ln_mlp_ratio,
                ln_no_adaptation=self.ln_no_adaptation,
                circumvent_tokens=int(self.cls_token_type is not None) +
                    int(self.distil_token_type is not None),
                kwargs=self.upsample_kwargs,
                name=f"{self.name_prefix}depth_{key}_{self.US[2:]}",
            )

        # Build this decoder depth's flatten or unflatten reshaper.
        if key in self.reshaper_ids_dict:
            stage[self.R] = self._create_reshaper(
                ids_dict=self.reshaper_ids_dict,
                i=i,
                layers_dicts=layers_dicts,
                layers_dict=stage,
                base_dim=self.dim,
                base_grid_size=self.grid_size,
                grid_has_tokens=int(self.cls_token_type is not None) +
                    int(self.distil_token_type is not None),
                kwargs=self.reshaper_kwargs,
                name=f"{self.name_prefix}depth_{key}_{self.R[2:]}",
            )

        # Build this decoder depth's auxiliary token head.
        if key in self.cls_token_regularizer_ids:
            stage[self.CTR] = self._create_token_regularizer(
                i=i, 
                layers_dicts=layers_dicts, 
                layers_dict=stage, 
                base_dim=self.dim, 
                kwargs=self.cls_token_regularizer_kwargs, 
                name=f"{self.name_prefix}depth_{key}_{self.CTR[2:]}"
            )

        return stage

    def get_config(self) -> dict[str, object]:
        """Return decoder architecture and standard Keras state.

        Returns:
            dict[str, object]: Serializable constructor configuration.
        """

        config = super().get_config()
        config.update({
            "name": self.name, 
            "trainable": self.trainable, 
            "dtype": self.dtype_policy.name, 
            "dynamic": self.dynamic, 
        })

        return config

    def _split_context_inputs(
        self, 
        inputs: tuple[tf.Tensor, ...], 
        encoder_cond: tf.Tensor | None, 
        encoder_features_list: list[tf.Tensor | None] | tuple[
            tf.Tensor | None, ...
        ] | None, 
    ) -> tuple[
        tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        tf.Tensor | None, 
        list[tf.Tensor | None], 
    ]:
        """Normalize packed and explicit decoder input forms.

        Args:
            inputs (tuple[tf.Tensor, ...]): Decoder tensors, optionally followed
                by encoder context.
            encoder_cond (tf.Tensor | None): Explicit encoder condition.
            encoder_features_list (list[tf.Tensor | None] | None): Explicit
                depth-indexed encoder features.

        Returns:
            tuple: Decoder inputs, condition, and feature list.

        Raises:
            ValueError: If arity or feature count is inconsistent.
        """

        # Unpack encoder features from the packed Keras functional input form.
        if encoder_features_list is None:
            # Require image, time, label, encoder condition, and routed features.
            if len(inputs) < 5:
                raise ValueError(
                    "packed decoder inputs must contain images, times, labels, "
                    "encoder_cond, and encoder features."
                )
            decoder_inputs = tuple(inputs[:3])
            encoder_cond = inputs[3]
            features = list(inputs[4:])
        # Use the explicit encoder-feature argument for direct eager calls.
        else:
            # Direct decoder inputs must contain image, time, and label tensors.
            if len(inputs) != 3:
                raise ValueError(
                    "decoder inputs must contain images, times, and labels."
                )
            decoder_inputs = tuple(inputs)
            features = list(encoder_features_list)
        # Keep runtime encoder features aligned with declared feature metadata.
        if len(features) != len(self.encoder_feature_dims):
            raise ValueError(
                "encoder_features_list must match encoder_feature_dims."
            )

        return decoder_inputs, encoder_cond, features

    def call(
        self, 
        inputs: tuple[tf.Tensor, ...], 
        encoder_cond: tf.Tensor | None = None, 
        encoder_features_list: list[tf.Tensor | None] | tuple[
            tf.Tensor | None, ...
        ] | None = None, 
        full_return: bool = False, 
        training: bool | None = None, 
        min_depth: int = 0, 
    ) -> dict[str, object]:
        """Predict noise using decoder inputs and encoder context.

        Args:
            inputs (tuple[tf.Tensor, ...]): Packed inputs or three decoder
                tensors used with explicit context arguments.
            encoder_cond (tf.Tensor | None): Encoder condition ``[B, E]``. Defaults to ``None``.
            encoder_features_list (list[tf.Tensor | None] | None): Encoder features matching
                ``encoder_feature_dims``. Defaults to ``None``.
            full_return (bool): Include standard intermediate values. Defaults to ``False``.
            training (bool | None): Keras execution mode: True enables training behavior such as dropout
                and normalization updates; False selects inference behavior; None inherits the enclosing
                Keras learning context. Variational sampling, when configured, remains active
                independently of this flag. Defaults to ``None``.
            min_depth (int): First decoder stage to execute. Defaults to ``0``.

        Returns:
            dict[str, object]: Noise output and optional intermediates.
        """

        decoder_inputs, encoder_cond, encoder_features = \
            self._split_context_inputs(
                inputs, encoder_cond, encoder_features_list
            )
        x, cond, features, regs, z_vals_list = self.decode(
            decoder_inputs, 
            encoder_cond, 
            encoder_features, 
            training=training, 
            min_depth=min_depth, 
            full_return=True, 
        )
        # Apply the decoder image head only when unpatchification is enabled.
        noises = self.unpatchifier((x, cond), training=training) \
                if self.use_unpatchify else x
        outputs = {"noises": noises}

        # Attach decoder condition and intermediate metadata only for full returns.
        if full_return:
            outputs.update({
                "cond": cond, 
                "features_list": features, 
                "regs_list": regs, 
                "z_vals_list": z_vals_list, 
                "decoder_cond": cond, 
                "decoder_features_list": features, 
                "encoder_cond": encoder_cond, 
                "encoder_features_list": encoder_features, 
            })
        return outputs

    def build(self, input_shape: tuple | None = None) -> None:
        """Build the packed decoder/context graph.

        Args:
            input_shape (tuple | None): Accepted by Keras and ignored. Defaults to ``None``.

        Returns:
            None: Variables are created in place.
        """

        shapes = self._build_model(call_model=True)
        tf.keras.Model.build(self, shapes)

    def _build_model(self, call_model: bool = True) -> list[tf.TensorShape]:
        """Create decoder inputs plus one input per encoder feature.

        Args:
            call_model (bool): Connect inputs through :meth:`call`. Defaults to ``True``.

        Returns:
            list[tf.TensorShape]: Shapes of every packed input.
        """

        DiffusionTransformer._build_model(self, call_model=False)
        decoder_inputs = self.inputs
        encoder_cond = layers.Input(
            shape=(self.cond_dim,), 
            dtype=self.compute_dtype,
            name="encoder_cond"
        )
        # Build rank-two symbolic inputs for flat features and rank-three inputs for token
        # features.
        encoder_features = tuple(
            layers.Input(
                shape=(dim,) if is_flat else (None, dim),
                dtype=self.compute_dtype,
                name=f"encoder_feature_{index}", 
            )
            for index, (dim, is_flat) in enumerate(zip(
                self.encoder_feature_dims, 
                self.encoder_feature_is_flat,
            ))
        )
        self.inputs = decoder_inputs + (encoder_cond,) + encoder_features
        # Execute the symbolic decoder graph only when construction requests a model call.
        self.outputs = self.call(self.inputs) if call_model else None

        return [input_layer.shape for input_layer in self.inputs]

    def get_causal_attention_mask(self, x: tf.Tensor) -> tf.Tensor:
        """Create the decoder self-attention mask for the current sequence.

        Args:
            x (tf.Tensor): Decoder tokens shaped ``[B, T, D]``.

        Returns:
            tf.Tensor: Lower-triangular boolean mask shaped ``[T, T]``. Token
            ``t`` may attend only to positions ``0..t``.
        """

        sequence_length = tf.shape(x)[1]

        return tf.linalg.band_part(
            tf.ones(
                (sequence_length, sequence_length), 
                dtype=tf.bool, 
            ), 
            -1, 
            0, 
        )

    def decode(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
        encoder_cond: tf.Tensor | None, 
        encoder_features_list: list[tf.Tensor | None] | tuple[
            tf.Tensor | None, ...
        ], 
        max_depth: int = -1, 
        full_return: bool = False, 
        training: bool | None = None, 
        min_depth: int = 0
    ) -> tuple:
        """Decode an image or an intermediate representation.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Decoder image,
                timestep IDs, and label IDs. At ``min_depth > 0``, the first
                tensor is the representation at that decoder depth.
            encoder_cond (tf.Tensor | None): Encoder condition ``[B, E]``.
                It is used when ``decoder_separate_cond=False``; ``None`` is
                replaced by a zero condition of width ``cond_dim``.
            encoder_features_list (list[tf.Tensor | None]): Encoder features
                indexed by ``encoder_feature_dims``. Aggregators select these
                tensors; without an explicit cross-attention route, the final
                non-``None`` feature is used as attention values.
            max_depth (int): Exclusive zero-based stage stop. ``-1`` executes every remaining stage and
                ``0`` executes no stage. Defaults to ``-1``.
            full_return (bool): Also return regularizer predictions and latent mean/log-variance values.
                Defaults to ``False``.
            training (bool | None): Keras execution mode: True enables training behavior such as dropout
                and normalization updates; False selects inference behavior; None inherits the enclosing
                Keras learning context. Variational sampling, when configured, remains active
                independently of this flag. Defaults to ``None``.
            min_depth (int): Number of initial decoder stages to skip. ``0`` embeds the decoder image;
                ``1..depth`` resumes from ``inputs[0]`` and fills skipped feature slots with ``None``.
                Defaults to ``0``.

        Returns:
            tuple: Normally ``(tokens, decoder_cond, features_list)``. With
            ``full_return=True``, returns ``(tokens, decoder_cond,
            features_list, regs_list, z_vals_list)``. Each ``z_vals_list`` item is one
            mean/log-variance pair. Returned tokens
            exclude optional class and distillation tokens, while retained
            features keep them in class, distillation, patch order.

        Raises:
            AssertionError: If ``min_depth`` is outside ``0..depth``.
        """

        require(0 <= min_depth <= self.depth, (
            "min_depth must be in the range of [0, depth]."
        ))


        decoder_input, times, labels = inputs
        # Normalize resumed decoding into one initial feature and any later supplied latents.
        latent_inputs = list(decoder_input) if min_depth > 0 and isinstance(
            decoder_input, (list, tuple)
        ) else [decoder_input]
        # Validate the latent-input count only when starting after the decoder entrance.
        if min_depth > 0:
            # Count only future flatten boundaries that require independently supplied latents.
            expected_latents = 1 + sum(
                flatten_id > min_depth and (
                    max_depth < 0 or flatten_id <= max_depth
                )
                for flatten_id, reshape_type in self.reshaper_ids_dict.items()
                if reshape_type == "flatten"
            )
            # Reject resumed inputs that omit or add a required later bottleneck latent.
            if len(latent_inputs) != expected_latents:
                raise ValueError(
                    f"Resuming at depth {min_depth} requires "
                    f"{expected_latents} input feature/latent tensors."
                )
        batch_input = latent_inputs[0]
        # Build decoder-owned conditions when separate conditioning is enabled.
        if self.decoder_separate_cond:
            cond, time_embeds, label_embeds = self.embed_conditions(
                times, 
                labels, 
                self.cond_type, 
                full_return=True, 
                training=training, 
            )
        # Otherwise reuse the condition produced by the encoder.
        else:
            cond = encoder_cond
            time_embeds = None
            label_embeds = None

        # Supply neutral conditioning when both encoder and decoder omit conditions.
        if cond is None:
            cond = tf.zeros(
                (tf.shape(batch_input)[0], self.cond_dim),
                dtype=self.compute_dtype, 
            )

        # Embed raw decoder images when execution starts at depth zero.
        if min_depth == 0:
            # Patchify at the active resolution only when it differs from native decoder
            # resolution.
            x = self.patch_embedder(
                decoder_input, 
                output_grid_size=(
                    self._current_resolution // self.patch_size
                    if self.image_size != self._current_resolution else None
                ), 
                training=training
            )
            # Merge the decoder condition into patch tokens when configured.
            if self.patches_conds_merger is not None:
                x = self.patches_conds_merger((
                    x, 
                    tf.repeat(
                        cond[:, None], 
                        tf.shape(x)[1], 
                        axis=1
                    )
                ), training=training)
            # Prepend distillation first so a class token remains position zero.
            if self.distil_token_type is not None:
                x = self.prepend_single_token(
                    x, self.distil_token, 
                    self.distil_token_type, 
                    time_embeds=time_embeds, 
                    label_embeds=label_embeds, 
                    times=times, 
                    labels=labels, 
                    training=training, 
                )
            # Prepend the configured decoder class token.
            if self.cls_token_type is not None:
                x = self.prepend_single_token(
                    x, self.cls_token, 
                    self.cls_token_type, 
                    time_embeds=time_embeds, 
                    label_embeds=label_embeds, 
                    times=times, 
                    labels=labels, 
                    training=training, 
                )
        # Resume from an already embedded decoder feature.
        else:
            x = batch_input

        # Compute labels solely for a depth-zero label regularizer when needed.
        if label_embeds is None and self.label_embedder is not None and \
        self.labels_embed_reg is not None:
            label_embeds = self.label_embedder(labels, training=training)
        # Compute depth-zero decoder label regularization only when its head exists.
        depth_zero_reg = self.labels_embed_reg(
            label_embeds, 
            training=training, 
        ) if self.labels_embed_reg is not None else None

        features_list = [None] * min_depth + [x]
        regs_list = [depth_zero_reg] + [None] * min_depth
        z_vals_list = []
        latent_index = 1
        for i, layers_dict in enumerate(self.layers_dicts):
            # Stop before the exclusive maximum decoder depth.
            if i == max_depth:
                break
            # Leave earlier decoder stages untouched during resumed execution.
            if i < min_depth:
                continue

            is_flatten = self.reshaper_ids_dict.get(i + 1) == "flatten"
            # During prior decoding, replace every later posterior-building
            # flatten stage before it can read unavailable encoder features.
            if self.R in layers_dict and min_depth > 0 and is_flatten:
                x = latent_inputs[latent_index]
                latent_index += 1
                # Regularize an injected latent only when this decoder stage owns an auxiliary
                # head.
                reg = layers_dict[self.CTR](
                    self.slice_and_flatten_tokens(
                        x,
                        self.cls_token_regularizer_kwargs["start"],
                        self.cls_token_regularizer_kwargs["end"]
                    ),
                    training=training
                ) if self.CTR in layers_dict else None
                features_list.append(x)
                regs_list.append(reg)
                continue

            # Aggregate routed encoder features into the decoder stream.
            if self.FA in layers_dict:
                # Append the current decoder stream unless a separate self-connector already
                # supplies it.
                x = layers_dict[self.FA](
                    encoder_features_list, 
                    [x] if self.FC not in layers_dict else [], 
                    cond=cond, 
                    training=training, 
                )

            # Merge routed earlier decoder features into the current stream.
            if self.FC in layers_dict:
                # Pass encoder aggregation into a self-connector only when that aggregate
                # exists.
                x = layers_dict[self.FC](
                    features_list, 
                    [x] if self.FA in layers_dict else [], 
                    cond=cond, 
                    training=training,
                )

            # Build external attention features from the encoder only at configured depths.
            h = layers_dict[self.CAA](
                encoder_features_list, 
                cond=cond, 
                training=training, 
            ) if self.CAA in layers_dict else None

            # Apply decoder self-attention routing when configured; otherwise retain the encoder
            # aggregate.
            # Append encoder attention features only when the cross-aggregator produced them.
            h = layers_dict[self.CAC](
                features_list, 
                [h] if self.CAA in layers_dict else [], 
                cond=cond, 
                training=training, 
            ) if self.CAC in layers_dict else h

            # Apply this depth's encoder or cross-attention transformer block.
            if self.VTB in layers_dict:
                block = layers_dict[self.VTB]
                # Send external attention features to queries only in query-side mode.
                queries = h if self.cross_attention_plug_type == "queries" else None
                # Send external attention features to keys/values only in value-side mode.
                values = h if self.cross_attention_plug_type == "values" else None
                # Fall back to the first available encoder feature as attention values.
                if h is None:
                    # Ignore unavailable encoder feature slots when finding a usable attention
                    # source.
                    values = next(
                        (
                            feature for feature in reversed(
                                encoder_features_list
                            )
                            if feature is not None
                        ), 
                        None, 
                    )

                block_kwargs = {
                    "queries": queries, 
                    "values": values, 
                    "training": training, 
                }
                # Supply a causal mask only to decoder-style attention blocks.
                if isinstance(block, DiTDecoderBlock):
                    # Build a lower-triangular mask only when causal decoder attention is
                    # enabled.
                    block_kwargs["causal_mask"] = (
                        self.get_causal_attention_mask(x)
                        if self.use_causal_mask else None
                    )
                x = block((x, cond), **block_kwargs)

            # Apply the decoder's local mixer only at selected stages.
            x = layers_dict[self.LM](
                (x, cond), training=training
            ) if self.LM in layers_dict else x

            # Reduce spatial resolution only at decoder downsample stages.
            x = layers_dict[self.DS](
                (x, cond), training=training
            ) if self.DS in layers_dict else x

            # Increase spatial resolution only at decoder upsample stages.
            x = layers_dict[self.US](
                (x, cond), training=training
            ) if self.US in layers_dict else x

            # Apply a decoder reshaper when present and leave absent latent statistics unset.
            x, x_mean, x_log_var = layers_dict[self.R](
                x, training=training
            ) if self.R in layers_dict else (x, None, None)
            # Evaluate a decoder auxiliary class head only when configured.
            reg = layers_dict[self.CTR](
                self.slice_and_flatten_tokens(
                    x, 
                    self.cls_token_regularizer_kwargs["start"], 
                    self.cls_token_regularizer_kwargs["end"], 
                ),
                training=training,
            ) if self.CTR in layers_dict else None

            features_list.append(x)
            regs_list.append(reg)
            # Preserve latent statistics emitted by a flattening reshaper.
            if x_mean is not None and is_flatten and bool(
                self.reshaper_kwargs.get("add_kl", False)
            ):
                z_vals_list.append((x_mean, x_log_var))

        # Remove both prefix tokens only from the returned decoder patch stream.
        prefix_tokens_num = int(self.cls_token_type is not None) + \
            int(self.distil_token_type is not None)
        # Preserve the established no-slice path when no prefix token exists.
        if prefix_tokens_num:
            x = x[:, prefix_tokens_num:]

        # Return decoder regularizers and latent values only on request.
        if full_return:
            return x, cond, features_list, regs_list, z_vals_list
        return x, cond, features_list

    def encode(
        self,
        inputs: tuple[tf.Tensor, ...], 
        encoder_cond: tf.Tensor | None = None, 
        encoder_features_list: list[tf.Tensor | None] | tuple[
            tf.Tensor | None, ...
        ] | None = None, 
        max_depth: int = -1, 
        training: bool | None = None, 
        min_depth: int = 0, 
    ) -> tuple[
        tf.Tensor, 
        tf.Tensor, 
        list[tf.Tensor | None], 
        list[tf.Tensor | None], 
        list[tuple[tf.Tensor, tf.Tensor]],
    ]:
        """Return the standard five-part decoder representation.

        Args:
            inputs (tuple[tf.Tensor, ...]): Explicit three-tensor decoder input
                or packed ``(image, time, label, encoder_cond, *features)``.
            encoder_cond (tf.Tensor | None): Explicit encoder condition. Defaults to ``None``.
            encoder_features_list (list[tf.Tensor | None] | None): Explicit encoder features. Omitting
                it selects the packed input form. Defaults to ``None``.
            max_depth (int): Exclusive stage stop forwarded to :meth:`decode`. Defaults to ``-1``.
            training (bool | None): Keras execution mode: True enables training behavior such as dropout
                and normalization updates; False selects inference behavior; None inherits the enclosing
                Keras learning context. Variational sampling, when configured, remains active
                independently of this flag. Defaults to ``None``.
            min_depth (int): First decoder depth to execute. Defaults to ``0``.

        Returns:
            tuple: ``(tokens, decoder_cond, decoder_features, regs_list,
            z_vals_list)``.
        """

        decoder_inputs, encoder_cond, encoder_features = \
            self._split_context_inputs(
                inputs, 
                encoder_cond, 
                encoder_features_list, 
            )

        return self.decode(
            decoder_inputs, 
            encoder_cond, 
            encoder_features, 
            max_depth=max_depth, 
            full_return=True, 
            training=training, 
            min_depth=min_depth, 
        )

    def predict_noise(
        self, 
        inputs: tuple[tf.Tensor, ...], 
        encoder_cond: tf.Tensor | None = None, 
        encoder_features_list: list[tf.Tensor | None] | tuple[
            tf.Tensor | None, ...
        ] | None = None, 
        full_return: bool = False, 
        training: bool | None = None, 
        min_depth: int = 0, 
    ) -> tf.Tensor | tuple:
        """Run the decoder noise path with explicit or packed context.

        Args:
            inputs (tuple[tf.Tensor, ...]): Input form accepted by
                :meth:`call`.
            encoder_cond (tf.Tensor | None): Explicit encoder condition. Defaults to ``None``.
            encoder_features_list (list[tf.Tensor | None] | None): Explicit encoder features; omit for
                packed inputs. Defaults to ``None``.
            full_return (bool): Return the standard transformer five-tuple instead of only the predicted
                noise. Defaults to ``False``.
            training (bool | None): Keras execution mode: True enables training behavior such as dropout
                and normalization updates; False selects inference behavior; None inherits the enclosing
                Keras learning context. Variational sampling, when configured, remains active
                independently of this flag. Defaults to ``None``.
            min_depth (int): First decoder stage to execute. Defaults to ``0``.

        Returns:
            tf.Tensor | tuple: Predicted image/noise, or ``(noises, cond,
            features_list, regs_list, z_vals_list)`` when ``full_return=True``.
        """

        outputs = self.call(
            inputs,
            encoder_cond=encoder_cond,
            encoder_features_list=encoder_features_list,
            full_return=full_return,
            training=training,
            min_depth=min_depth,
        )

        # Preserve the extended decoder output tuple only for full returns.
        if full_return:
            return (
                outputs["noises"], 
                outputs["cond"], 
                outputs["features_list"], 
                outputs["regs_list"], 
                outputs["z_vals_list"], 
            )
        return outputs["noises"]

    def set_encoder_feature_metadata(
        self, 
        encoder_feature_dims: list[int] | tuple[int, ...], 
        encoder_feature_grid_sizes: list[int | None] | tuple[
            int | None, ...
        ],
        encoder_feature_is_flat: list[bool] | tuple[bool, ...] | None = None,
    ) -> None:
        """Extend encoder metadata after progressive encoder growth.

        Existing indices are immutable because constructed aggregation layers
        depend on their widths and grids. New trailing entries become
        available to subsequently added decoder depths.

        Args:
            encoder_feature_dims (list[int]): Positive feature widths for all
                encoder depths, including depth 0.
            encoder_feature_grid_sizes (list[int | None]): Matching spatial
                grid sides. ``None`` may denote a flat feature or non-square
                tokens; the final entry must be spatial. Its width must stay
                stable for existing attention projections, while its token
                grid may change.
            encoder_feature_is_flat (list[bool] | None): Matching explicit rank states. ``None`` uses
                the legacy grid-based inference. Defaults to ``None``.

        Returns:
            None: Metadata and serialized constructor state are updated.

        Raises:
            ValueError: If the sequences differ in length, shrink, alter an
                existing entry, contain invalid values, or end in a flat
                feature.
        """

        dims = [int(dim) for dim in encoder_feature_dims]
        # Normalize numeric encoder grids while preserving None for nonspatial features.
        grids = [
            None if grid is None else int(grid)
            for grid in encoder_feature_grid_sizes
        ]
        # Use supplied encoder rank metadata or infer it from the grid metadata.
        flat_states = list(encoder_feature_is_flat) if \
            encoder_feature_is_flat is not None else [
                grid is None for grid in grids
            ]
        # Require nonempty, aligned encoder width and grid metadata.
        if not dims or len(dims) != len(grids) or len(dims) != len(flat_states):
            raise ValueError(
                "encoder feature dimensions, grids, and rank states must be "
                "non-empty and have equal length."
            )
        # Reject nonpositive feature widths and spatial grid sides in updated metadata.
        if any(dim < 1 for dim in dims) or any(
            grid is not None and grid < 1 for grid in grids
        ):
            raise ValueError(
                "encoder feature dimensions and supplied grids must be positive."
            )
        # Progressive metadata may extend but never remove existing features.
        if len(dims) < len(self.encoder_feature_dims):
            raise ValueError("encoder feature metadata cannot shrink.")
        old_count = len(self.encoder_feature_dims)
        # Preserve metadata for every previously registered encoder feature.
        if dims[:old_count] != self.encoder_feature_dims or \
        grids[:old_count] != self.encoder_feature_grid_sizes or \
        flat_states[:old_count] != self.encoder_feature_is_flat:
            raise ValueError("existing encoder feature metadata is immutable.")
        # Keep the final encoder output spatial for decoder conditioning.
        if grids[-1] is None or flat_states[-1]:
            raise ValueError("the final encoder feature must have a spatial grid.")
        # Keep progressive encoder output width compatible with decoder construction.
        if dims[-1] != self.encoder_output_dim:
            raise ValueError(
                "progressive encoder metadata must preserve the final width "
                "used by existing decoder blocks."
            )

        self.encoder_feature_dims = dims
        self.encoder_feature_grid_sizes = grids
        self.encoder_feature_is_flat = flat_states
        self.encoder_output_dim = dims[-1]
        self.encoder_output_grid_size = grids[-1]
        self._init_config.update({
            "encoder_feature_dims": deepcopy(dims), 
            "encoder_feature_grid_sizes": deepcopy(grids), 
            "encoder_feature_is_flat": deepcopy(flat_states),
            "encoder_output_dim": self.encoder_output_dim, 
            "encoder_output_grid_size": self.encoder_output_grid_size, 
            "feature_aggregation_ids_dict": deepcopy(
                self.feature_aggregation_ids_dict
            ), 
            "cross_attention_aggregation_ids_dict": deepcopy(
                self.cross_attention_aggregation_ids_dict
            ), 
        })

    def _apply_depths(
        self, 
        depth_spec: str | tuple | set | dict | list | None, 
    ) -> dict[str, dict[str, int]]:
        """Apply a validated decoder growth specification.

        The base progressive names remain valid. Two decoder-only names are
        added: ``feature_aggregator`` selects encoder features for the decoder
        stream, and ``cross_attention_aggregator`` selects encoder features for
        attention. Their value may be an ID, an ID sequence, ``True``/``None``
        for the final encoder feature, or ``{"ids": ...}``. Newly added
        transformer blocks default to ``DiTDecoderBlock``; specify
        ``{"use_decoder": False}`` for an encoder-style block.

        Args:
            depth_spec (str | tuple | set | dict | list | None): One stage or a
                list of stage specifications. Example:
                ``{"feature_aggregator": {"ids": [0, -1]},
                "vision_transformer_block": True}``.

        Returns:
            dict[str, dict[str, int]]: Standard ``network`` growth counts.

        Raises:
            ValueError: If a layer name is unknown or output-head shape would
                change. Encoder IDs outside the metadata range raise
                ``AssertionError``.
        """

        # Normalize one decoder growth description or an explicit list of them.
        depth_specs = depth_spec if isinstance(depth_spec, list) else [depth_spec]
        # Ignore disabled decoder growth placeholders.
        depth_specs = [spec for spec in depth_specs if spec is not None]
        old_feature_ids = deepcopy(self.feature_aggregation_ids_dict)
        old_cross_ids = deepcopy(self.cross_attention_aggregation_ids_dict)
        prepared_specs = []

        try:
            for offset, raw_spec in enumerate(depth_specs):
                # Interpret a string as one enabled decoder layer.
                if isinstance(raw_spec, str):
                    layer_spec = {raw_spec: True}
                # Interpret a collection as several enabled layers in one depth.
                elif isinstance(raw_spec, (tuple, set, frozenset)):
                    layer_spec = dict.fromkeys(raw_spec, True)
                # Copy a mapped decoder depth specification for normalization.
                else:
                    layer_spec = dict(raw_spec)

                key = self.depth + offset + 1
                for layer_name, mapping_name in (
                    (self.FA[2:], "feature_aggregation_ids_dict"), 
                    (self.CAA[2:], "cross_attention_aggregation_ids_dict"), 
                ):
                    # Leave absent feature-handler types unregistered.
                    if layer_name not in layer_spec:
                        continue
                    options = layer_spec.pop(layer_name)
                    # Ignore explicitly disabled feature handlers.
                    if options is False:
                        continue
                    # Copy explicit decoder aggregation settings before inserting a new stage.
                    ids = options.get("ids") \
                        if isinstance(options, dict) else options
                    # Use the final encoder feature when a new aggregation source is
                    # unspecified.
                    ids = -1 if ids is None or ids is True else ids
                    setattr(self, mapping_name, {
                        **getattr(self, mapping_name),
                        key: self._normalize_encoder_ids(ids),
                    })

                block_name = self.VTB[2:]
                # Normalize options for an enabled transformer block.
                if block_name in layer_spec and layer_spec[block_name] is not False:
                    block_options = layer_spec[block_name]
                    # Copy explicit decoder block settings or use defaults for shorthand blocks.
                    block_options = dict(block_options) \
                        if isinstance(block_options, dict) else {}
                    block_options.setdefault("use_decoder", True)
                    layer_spec[block_name] = block_options
                prepared_specs.append(layer_spec)

            growth = DiffusionTransformer.add_depths(self, prepared_specs)
        except Exception:
            self.feature_aggregation_ids_dict = old_feature_ids
            self.cross_attention_aggregation_ids_dict = old_cross_ids
            raise

        self._init_config.update({
            "feature_aggregation_ids_dict": deepcopy(
                self.feature_aggregation_ids_dict
            ),
            "cross_attention_aggregation_ids_dict": deepcopy(
                self.cross_attention_aggregation_ids_dict
            ),
        })

        return growth

    def add_depths(
        self, 
        depth_spec: str | tuple | set | dict | list | None, 
    ) -> dict[str, dict[str, int]]:
        """Append decoder stages without invalidating the existing head.

        Base transformer layer names and the decoder-only
        ``feature_aggregator``/``cross_attention_aggregator`` names are valid.
        New transformer blocks default to :class:`DiTDecoderBlock`. The change
        is first applied to an unbuilt clone, so invalid IDs, layer names,
        widths, or final token grids leave this decoder unchanged.

        Args:
            depth_spec (str | tuple | set | dict | list | None): One stage or a
                list of stage specifications using the syntax documented by
                :meth:`DiffusionTransformer.add_depths`.

        Returns:
            dict[str, dict[str, int]]: Standard ``network`` growth counts.

        Raises:
            ValueError: If the specification is invalid or changes the shape
                expected by the existing output head.
        """

        old_grid = self._get_last_grid_size(
            self.depth - 1, self.layers_dicts, self.grid_size
        )
        probe_config = self.get_config()
        probe_config["build"] = False
        probe = DiTDecoder.from_config(probe_config)
        probe._apply_depths(deepcopy(depth_spec))
        new_grid = probe._get_last_grid_size(
            probe.depth - 1, probe.layers_dicts, probe.grid_size
        )
            # Keep progressive token geometry compatible with the existing image head.
        if self.use_unpatchify and new_grid != old_grid:
            raise ValueError(
                "Added depths must preserve the output-head token grid."
            )

        return self._apply_depths(depth_spec)


def run_self_tests() -> dict[str, str]:
    """Test decoder construction, context routing, execution, and growth.

    Returns:
        dict[str, str]: One passed entry for :class:`DiTDecoder`.

    The checks construct small TensorFlow models, reset Keras session state, and
    seed random streams. Successful completion returns the named pass mapping;
    failed numerical, shape, serialization, or invalid-input expectations raise.

    Raises:
        AssertionError: If a regression expectation fails.
        tf.errors.InvalidArgumentError: If a TensorFlow numerical assertion fails.
    """

    import numpy as np


    common = {
        "encoder_output_grid_size": 2, 
        "encoder_output_dim": 4, 
        "encoder_feature_grid_sizes": [2, 2], 
        "encoder_feature_dims": [4, 4], 
        "num_classes": 2, 
        "timesteps": 8, 
        "image_size": 4, 
        "channels": 1, 
        "patch_size": 2, 
        "dim": 4, 
        "mha_num_heads": 1, 
        "vit_block_mlp_ratio": 1.0, 
        "shift_inputs": False, 
        "build": False, 
    }
    decoder = DiTDecoder(
        depth=1, 
        feature_aggregation_ids_dict={1: [0]}, 
        feature_aggregation_kwargs={"connect_type": "add"}, 
        cross_attention_aggregation_ids_dict={1: [-1]}, 
        cross_attention_aggregation_kwargs={"connect_type": "add"}, 
        **common,
    )
    assert decoder.use_decoder_ids == [1]
    assert set(decoder.layers_dicts[0]) == {decoder.FA, decoder.CAA, decoder.VTB}

    images = tf.reshape(tf.range(32, dtype=tf.float32), (2, 4, 4, 1)) / 32
    times = tf.constant([1, 2], tf.int32)
    labels = tf.constant([1, 2], tf.uint8)
    encoder_cond = tf.ones((2, 4))
    encoder_features = [
        tf.ones((2, 4, 4)), 
        tf.fill((2, 4, 4), 2.0), 
    ]
    legacy = decoder(
        (images, times, labels), 
        encoder_cond, encoder_features, 
        full_return=True, training=False, 
    )
    packed = decoder(
        (images, times, labels, encoder_cond, *encoder_features), 
        full_return=True, training=False,
    )
    assert set(legacy) == {
        "noises", "cond", "features_list", "regs_list", "z_vals_list", 
        "decoder_cond", "decoder_features_list", "encoder_cond", 
        "encoder_features_list", 
    }
    np.testing.assert_allclose(legacy["noises"], packed["noises"], atol=1e-6)
    assert legacy["noises"].shape == (2, 4, 4, 1)
    assert len(legacy["features_list"]) == 2
    assert len(legacy["regs_list"]) == 2
    noise_full = decoder.predict_noise(
        (images, times, labels), encoder_cond, encoder_features, 
        full_return=True, training=False, 
    )
    assert len(noise_full) == 5 and noise_full[0].shape == (2, 4, 4, 1)
    encoded = decoder.encode(
        (images, times, labels), encoder_cond, encoder_features, 
        training=False, 
    )
    assert len(encoded) == 5 and encoded[0].shape == (2, 4, 4)

    symbolic_shapes = decoder._build_model(call_model=True)
    assert len(symbolic_shapes) == 6
    assert len(decoder.inputs) == 6 and set(decoder.outputs) == {"noises"}

    growth = decoder.add_depths({
        "feature_aggregator": {"ids": [-1]}, 
        "cross_attention_aggregator": {"ids": [0]}, 
        "vision_transformer_block": True, 
    })
    assert growth["network"] == {"before": 1, "added": 1, "after": 2}
    assert isinstance(decoder.layers_dicts[-1][decoder.VTB], DiTDecoderBlock)
    assert decoder.feature_aggregation_ids_dict[2] == [1]
    assert decoder.cross_attention_aggregation_ids_dict[2] == [0]
    assert decoder((images, times, labels), encoder_cond, encoder_features)[
        "noises"
    ].shape == (2, 4, 4, 1)

    final_feature_growth = DiTDecoder(
        depth=0, 
        feature_aggregation_kwargs={"connect_type": "add"}, 
        **common,
    )
    final_feature_growth.add_depths({"feature_aggregator": None})
    assert final_feature_growth.feature_aggregation_ids_dict == {1: [1]}
    assert final_feature_growth.FA in final_feature_growth.layers_dicts[0]

    feature_axis = DiTDecoder(
        depth=1, 
        feature_aggregation_ids_dict={1: [0, 1]}, 
        feature_aggregation_kwargs={"connect_axis": -1},
        dim_forced=False, 
        use_unpatchify=False, 
        **common,
    )
    feature_axis_output = feature_axis(
        (images, times, labels), encoder_cond, encoder_features,
    )["noises"]
    assert feature_axis.layers_dicts[0][feature_axis.FA].output_dim == 12
    assert feature_axis_output.shape == (2, 4, 12)

    connector_features = DiTDecoder(
        depth=1, 
        feature_aggregation_ids_dict={1: [0]}, 
        connection_ids_dict={1: [0]}, 
        feature_aggregation_kwargs={"connect_axis": -1},
        connection_kwargs={"connect_axis": -1},
        vit_block_ids=[], 
        use_decoder_ids=[], 
        dim_forced=False, 
        use_unpatchify=False, 
        **common,
    )
    connector_output = connector_features(
        (images, times, labels), encoder_cond, encoder_features,
    )["noises"]
    assert connector_features.layers_dicts[0][connector_features.FC].output_dim == 8
    assert connector_output.shape.rank == 3
    assert connector_features._get_last_grid_size(
        0, connector_features.layers_dicts, connector_features.grid_size
    ) == connector_features.grid_size
    assert connector_output.shape == (2, 4, 8)

    # Let this fixture infer encoder metadata from its changed final-feature shape.
    unknown_grid_features = DiTDecoder(
        depth=1,
        vit_block_ids=[],
        use_decoder_ids=[],
        feature_aggregation_ids_dict={1: [0]},
        feature_aggregation_kwargs={"connect_axis": -1},
        use_unpatchify=False,
        encoder_feature_grid_sizes=[None, 2],
        encoder_feature_dims=[4, 4],
        encoder_feature_is_flat=[False, False],
        **{key: value for key, value in common.items() if key not in (
            "encoder_feature_grid_sizes", "encoder_feature_dims"
        )},
    )
    unknown_grid_handler = unknown_grid_features.layers_dicts[0][
        unknown_grid_features.FA
    ]
    assert unknown_grid_handler.grid_size is None
    assert unknown_grid_handler.output_is_flat is False
    unknown_grid_output = unknown_grid_features(
        (images, times, labels),
        encoder_cond,
        [tf.ones((2, 4, 4)), encoder_features[-1]],
    )["noises"]
    assert unknown_grid_output.shape == (2, 4, 4)

    try:
        # Remove previous encoder metadata before constructing the malformed-context fixture.
        DiTDecoder(
            depth=1, 
            cross_attention_aggregation_ids_dict={1: [0]}, 
            cross_attention_plug_type="queries", 
            encoder_feature_grid_sizes=[3, 2], 
            encoder_feature_dims=[4, 4], 
            **{key: value for key, value in common.items() if key not in (
                "encoder_feature_grid_sizes", "encoder_feature_dims"
            )},
        )
    except AssertionError:
        pass
    # Fail this regression if no exception occurs: Growth must preserve the existing output
    # grid.
    else:
        raise AssertionError("Cross-attention queries must match decoder tokens.")

    grid_guard = DiTDecoder(depth=0, **common)
    try:
        grid_guard.add_depths("downsampler")
    except ValueError:
        pass
    # Fail this regression if no exception occurs: Growth must preserve the existing output
    # grid.
    else:
        raise AssertionError("Growth must preserve the existing output grid.")
    assert grid_guard.depth == 0 and not grid_guard.layers_dicts

    decoder.set_encoder_feature_metadata([4, 4, 4], [2, 2, 2])
    assert decoder.encoder_output_dim == 4
    try:
        decoder.set_encoder_feature_metadata([8, 4, 4], [2, 2, 2])
    except ValueError:
        pass
    # Fail this regression if no exception occurs: Existing encoder metadata must be immutable.
    else:
        raise AssertionError("Existing encoder metadata must be immutable.")
    try:
        decoder.set_encoder_feature_metadata([4, 4, 4, 8], [2, 2, 2, 2])
    except ValueError:
        pass
    # Fail this regression if no exception occurs: Progressive metadata must preserve final
    # shape.
    else:
        raise AssertionError("Progressive metadata must preserve final shape.")

    # Regenerate encoder metadata for the alternate routing fixture.
    bottleneck = DiTDecoder(
        depth=2, 
        vit_block_ids=[], 
        use_decoder_ids=[], 
        reshaper_ids_dict={1: "flatten", 2: "unflatten"}, 
        reshaper_kwargs={"add_kl": True, "latent_dim_ratio": [0.5]},
        cls_token_regularizer_ids=[2], 
        encoder_feature_grid_sizes=[2], 
        encoder_feature_dims=[4], 
        **{key: value for key, value in common.items() if key not in (
            "encoder_feature_grid_sizes", "encoder_feature_dims"
        )}, 
    )
    bottleneck_full = bottleneck.predict_noise(
        (images, times, labels), encoder_cond, [encoder_features[-1]],
        full_return=True, training=False,
    )
    assert bottleneck_full[0].shape == (2, 4, 4, 1)
    assert bottleneck_full[3][-1].shape == (2, 2)
    assert bottleneck_full[4][0][0].shape[0] == 2
    assert bottleneck_full[4][0][1].shape == bottleneck_full[4][0][0].shape
    assert bottleneck._get_last_grid_size(
        0, bottleneck.layers_dicts, bottleneck.grid_size
    ) == 0
    assert bottleneck._get_last_grid_size(
        1, bottleneck.layers_dicts, bottleneck.grid_size
    ) == 2

    multilevel = DiTDecoder(
        depth=8,
        vit_block_ids=[],
        use_decoder_ids=[],
        feature_aggregation_ids_dict={4: [0], 6: [1]},
        reshaper_ids_dict={
            2: "flatten", 3: "unflatten",
            4: "flatten", 5: "unflatten",
            6: "flatten", 7: "unflatten",
        },
        reshaper_kwargs={
            "add_kl": True,
            "latent_dim_ratio": [0.5, 1.0, 0.25],
        },
        **common,
    )
    multilevel_full = multilevel.predict_noise(
        (images, times, labels),
        encoder_cond,
        encoder_features,
        full_return=True,
        training=False,
    )
    assert [
        int(z_mean.shape[-1])
        for z_mean, _ in multilevel_full[-1]
    ] == [8, 16, 4]
    resumed_multilevel = multilevel.predict_noise(
        ([tf.zeros((2, 16))] * 3, times, labels),
        None,
        [None, None],
        min_depth=2,
        training=False,
    )
    assert resumed_multilevel.shape == (2, 4, 4, 1)
    truncated_multilevel = multilevel.decode(
        ([tf.zeros((2, 16))], times, labels),
        None,
        [None, None],
        min_depth=2,
        max_depth=3,
        training=False,
    )
    assert truncated_multilevel[0].shape == (2, 4, 4)

    # Regenerate encoder metadata for the independent decoder-context fixture.
    depth_zero_reg = DiTDecoder(
        depth=0, 
        cls_token_regularizer_ids=[0], 
        encoder_feature_grid_sizes=[2], 
        encoder_feature_dims=[4], 
        **{key: value for key, value in common.items() if key not in (
            "encoder_feature_grid_sizes", "encoder_feature_dims"
        )},
    )
    assert depth_zero_reg.label_embedder is not None
    zero_full = depth_zero_reg.predict_noise(
        (images, times, labels), encoder_cond, [encoder_features[-1]], 
        full_return=True, 
    )
    assert zero_full[3][0].shape == (2, 2)

    # Remove base encoder metadata so the focused fixture supplies its own dimensions.
    all_regularizers = DiTDecoder(
        depth=1, 
        cls_token_type="new_weight", 
        cls_token_regularizer_ids=[0, 1], 
        encoder_feature_grid_sizes=[2], 
        encoder_feature_dims=[4], 
        **{key: value for key, value in common.items() if key not in (
            "encoder_feature_grid_sizes", "encoder_feature_dims"
        )}, 
    )
    all_regs = all_regularizers.predict_noise(
        (images, times, labels), encoder_cond, [encoder_features[-1]], 
        full_return=True, 
    )[3]
    assert len(all_regs) == 2
    assert all(reg.shape == (2, 2) for reg in all_regs)

    config = decoder.get_config()
    clone = DiTDecoder.from_config(config)
    assert clone.encoder_feature_dims == [4, 4, 4]
    assert clone.depth == 2 and clone.use_decoder_ids == [1, 2]
    assert clone.cross_attention_aggregation_ids_dict[1] == [1]
    assert clone.name == decoder.name

    for bad_kwargs in (
        {"feature_aggregation_kwargs": {"unknown": True}}, 
        {"cross_attention_aggregation_kwargs": {"unknown": True}}, 
    ):
        try:
            DiTDecoder(depth=1, **bad_kwargs, **common)
        except AssertionError:
            pass
        # Fail this regression if no exception occurs: Encoder feature counts must match
        # metadata.
        else:
            raise AssertionError("Unknown aggregation kwargs must fail.")
    try:
        decoder((images, times, labels), encoder_cond, [encoder_features[0]])
    except ValueError:
        pass
    # Fail this regression if no exception occurs: Encoder feature counts must match metadata.
    else:
        raise AssertionError("Encoder feature counts must match metadata.")

    tf.keras.backend.clear_session()
    return {"DiTDecoder": "passed"}


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
