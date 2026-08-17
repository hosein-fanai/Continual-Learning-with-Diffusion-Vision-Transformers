"""Experimental decoder-style specialization of the diffusion transformer.

``DiTDecoder`` consumes a shifted decoder image sequence together with condition
and feature tensors produced by an encoder.  It is the decoder half used by
``DiTEncoderDecoder``; the generic training/sampling wrappers remain in
``diffusion.models.wrapper``.
"""

import tensorflow as tf
from tensorflow.keras import layers

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer


class DiTDecoder(DiffusionTransformer):
    """Decode image tokens while attending to encoder-side representations.

    The class reuses ``DiffusionTransformer`` embedding, block, mixer, scaler,
    and routing helpers and adds conceptual slots for encoder feature and
    cross-attention aggregation.  Decoder depth 0 is the shifted patch-token
    input; depths 1..N are the inherited ordered processing stages.  Encoder
    feature lists follow the same convention: index 0 is encoder depth 0 and
    index k is encoder depth k.

    Note:
        This is a legacy/experimental interface.  In the current implementation
        ``feature_aggregation_ids_dict`` and
        ``cross_attention_aggregation_ids_dict`` are saved as configuration but
        no decoder-specific layer factory materializes their ``FA``/``CAA``
        slots.  ``build_model(call_model=True)`` also cannot supply the separate
        condition and feature-list arguments required by :meth:`call`.  The
        output path still calls the legacy name ``unpatchify`` although the base
        class creates ``unpatchifier``; the separate-condition/class-token path
        likewise retains an older call signature.  Use configurations that
        bypass those paths only after validation, or adapt the integration
        before relying on this class for training.
    """

    FA  = "0_feature_aggregator"
    FC  = "1_feature_connector"
    CAA = "2_cross_attention_aggregator"
    CAC = "3_cross_attention_connector"
    VTB = "4_vision_transformer_block"
    LM  = "5_local_mixer"
    DS  = "6_downsample"
    US  = "7_upsample"

    def __init__(
        self, 
        encoder_output_grid_size: int, 
        encoder_output_dim: int, 
        shift_inputs: bool = True, 
        use_decoder_ids: list[int | None] = [None], 
        decoder_separate_cond: bool = False, 
        use_causal_mask: bool = True, 
        feature_aggregation_ids_dict: dict[
            int, list[int | None] | tuple[int | None, ...]
        ] = {1: (-1,)},
        feature_aggregation_kwargs: dict = {}, 
        cross_attention_aggregation_ids_dict: dict[
            int, list[int | None] | tuple[int | None, ...]
        ] = {},
        cross_attention_aggregation_kwargs: dict = {}, 
        build: bool = True, 
        **kwargs
    ):
        """Initialize decoder configuration and optionally build the model.

        Args:
            encoder_output_grid_size (int): Side length G of the encoder token
                grid represented by the symbolic encoder input.
            encoder_output_dim (int): Encoder token feature width D.
            shift_inputs (bool): Right-shift decoder patch tokens by prepending a
                zero token; true by default for autoregressive teacher forcing.
            use_decoder_ids (list[int | None]): Depths implemented with causal-
                capable ``DiTDecoderBlock``. ``[None]`` expands to every decoder
                depth; ``[]`` selects encoder-style blocks.
            decoder_separate_cond (bool): Keep decoder-owned time/label
                embedders when true.  False clears those embedder attributes so
                callers are expected to provide ``encoder_cond``.
            use_causal_mask (bool): Supply a lower-triangular attention mask to
                decoder blocks.
            feature_aggregation_ids_dict (dict[int, list[int | None]]): Reserved
                mapping from target decoder depths to encoder feature IDs.
                ``{1: (-1,)}`` denotes the encoder's final depth at decoder
                depth 1; ``None`` denotes all eligible encoder depths.  The
                current class stores but does not materialize these aggregators.
            feature_aggregation_kwargs (dict[str, object]): Reserved
                ``FeatureHandler`` options: ``connect_axis`` (int),
                ``connect_type`` (``"concat"``/``"add"``),
                ``use_layer_norm`` (bool), ``ln_dim`` (int | None),
                ``ln_mlp_ratio`` (float | None), ``ln_no_adaptation`` (bool),
                ``mlp_output_dim`` (int | None), ``mlp_ratio`` (float | None),
                and ``mlp_activation_func`` (Keras activation).
            cross_attention_aggregation_ids_dict (dict[int, list[int | None]]):
                Reserved encoder-feature IDs for decoder cross attention, with
                the same depth syntax as ``feature_aggregation_ids_dict``.
            cross_attention_aggregation_kwargs (dict[str, object]): Reserved
                cross-attention ``FeatureHandler`` options with the same exact
                keys as ``feature_aggregation_kwargs``.
            build (bool): Invoke inherited symbolic build immediately.  Because
                this experimental class needs external structured inputs,
                callers may set false and build/call it explicitly.
            **kwargs: ``DiffusionTransformer`` arguments (for example ``depth``,
                connection ID mappings, block IDs, dimensions, and output-head
                options) plus standard Keras ``Model`` keys ``name``,
                ``trainable``, ``dtype``, and ``dynamic``.

        Returns:
            None.  Decoder layers and configuration are initialized in place.
        """
        super().__init__(
            shift_inputs=shift_inputs, 
            use_decoder_ids=use_decoder_ids, 
            build=False, 
            **kwargs
        )
        self._save_init_args(locals())

        if not self.decoder_separate_cond:
            self.time_embedder = None
            self.label_embedder = None
            self.conds_merger = None

        if self.build_:
            self.build()

    def call(
        self, 
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        encoder_cond: tf.Tensor | None,
        encoder_features_list: list[tf.Tensor],
        full_return: bool = False,
        training: bool | None = None
    ) -> dict[str, object]:
        """Decode a noisy/teacher-forcing input with encoder context.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Decoder images
                ``[B,H,W,C]``, integer timesteps ``[B]``, and labels ``[B]``.
            encoder_cond (tf.Tensor | None): Encoder condition ``[B, cond_dim]``.
            encoder_features_list (list[tf.Tensor]): Encoder token tensors
                indexed by depth.
            full_return (bool): Include conditions and both feature lists.
            training (bool | None): Keras training mode.

        Returns:
            dict[str, object]: Always contains ``"noises"`` with image-shaped
            output when unpatchification is enabled, otherwise final rank-3
            tokens.  Full return also adds ``decoder_cond``,
            ``decoder_features_list``, ``encoder_cond``, and
            ``encoder_features_list``.
        """
        x, decoder_cond, decoder_features_list = self.decode(
            inputs, encoder_cond, 
            encoder_features_list, 
            training=training
        )
        noises = self.unpatchify(
            x, decoder_cond, 
            training=training
        ) if self.use_unpatchify else x

        output_dict = {
            "noises": noises, 
        }
        if full_return:
            output_dict["decoder_cond"] = decoder_cond
            output_dict["decoder_features_list"] = decoder_features_list
            output_dict["encoder_cond"] = encoder_cond
            output_dict["encoder_features_list"] = encoder_features_list

        return output_dict

    def build_model(self, call_model: bool = True) -> list[tf.TensorShape]:
        """Create decoder and encoder-output symbolic input shapes.

        Args:
            call_model (bool): Attempt to connect symbolic inputs to
                :meth:`call`.  ``False`` only creates inputs; because ``call``
                requires separate condition and feature-list arguments, this is
                the usable mode for the current standalone implementation.

        Returns:
            list[tf.TensorShape]: Base image/time/label shapes plus encoder token
            shape ``[None, encoder_output_grid_size**2, encoder_output_dim]``.
        """
        super().build_model(
            call_model=False
        ) # TODO: problematic?

        encoder_outputs = layers.Input(
            shape=(
                self.encoder_output_grid_size * self.encoder_output_grid_size, 
                self.encoder_output_dim
            ),
            name="encoder_outputs"
        )

        self.inputs += (encoder_outputs,)
        self.outputs = self.call(self.inputs) if call_model else None

        input_shape = [
            input_layer.shape for input_layer in self.inputs
        ]

        return input_shape

    def get_causal_attention_mask(self, x: tf.Tensor) -> tf.Tensor:
        """Create a lower-triangular self-attention mask.

        Args:
            x (tf.Tensor): Token tensor ``[B, sequence_length, D]``.

        Returns:
            tf.Tensor: Boolean mask ``[sequence_length, sequence_length]`` where
            a query can attend only to its own and earlier positions.
        """
        seq_len = tf.shape(x)[1]
        mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=tf.bool),
            -1,
            0,
        )

        return mask

    def decode(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        encoder_cond: tf.Tensor | None,
        encoder_features_list: list[tf.Tensor],
        max_depth: int = -1,
        training: bool | None = None
    ) -> tuple[tf.Tensor, tf.Tensor | None, list[tf.Tensor]]:
        """Run decoder token processing without the output image head.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Decoder input image
                ``[B,H,W,C]``, timestep IDs ``[B]``, and label IDs ``[B]``.
            encoder_cond (tf.Tensor | None): Encoder condition vector ``[B,E]``.
            encoder_features_list (list[tf.Tensor]): Encoder features indexed by
                depth.  They are consumed only if an ``FA`` or ``CAA`` layer is
                present in a stage.
            max_depth (int): Exclusive zero-based stop; ``-1`` executes all
                stages and ``0`` stops before the first stage.
            training (bool | None): Keras training mode.

        Returns:
            tuple[tf.Tensor, tf.Tensor | None, list[tf.Tensor]]: Final tokens
            with any class token removed, the decoder condition, and decoder
            features whose index 0 is the embedded input.
        """
        decoder_input_images, ts, labels = inputs

        if self.cond_type == "":
            decoder_cond, _, decoder_label_embeds = self.embed_conditions(
                ts, labels, 
                full_return=True, 
                training=training
            )
        else:
            decoder_cond = encoder_cond
            decoder_label_embeds = ...

        x = self.patch_embedder(
            decoder_input_images, 
            training=training
        )
        x = self.prepend_cls_token(
            x, decoder_cond, 
            decoder_label_embeds, 
            training=training
        ) if self.cls_token_type is not None else x

        causal_mask = self.get_causal_attention_mask(
            x
        ) if self.use_causal_mask else None

        decoder_features_list = [x]
        for i, layers_dict in enumerate(self.layers_dicts):
            if i == max_depth:
                break

            x = layers_dict[self.FA](
                encoder_features_list, 
                [x] if self.FC not in layers_dict and i != 0 else [], 
                cond=decoder_cond, 
                training=training
            ) if self.FA in layers_dict else x

            x = layers_dict[self.FC](
                decoder_features_list, 
                [x] if self.FA in layers_dict else [], 
                cond=decoder_cond, 
                training=training
            ) if self.FC in layers_dict else x

            h = layers_dict[self.CAA](
                encoder_features_list, 
                cond=decoder_cond, 
                training=training
            ) if self.CAA in layers_dict else None

            h = layers_dict[self.CAC](
                decoder_features_list, 
                [h] if self.CAA in layers_dict else [], 
                cond=decoder_cond, 
                training=training
            ) if self.CAC in layers_dict else h

            x = layers_dict[self.VTB](
                (x, decoder_cond), 
                queries=h if self.cross_attention_plug_type == "queries" else None, 
                values=h if self.cross_attention_plug_type == "values" else None, 
                causal_mask=causal_mask, 
                training=training
            ) if self.VTB in layers_dict else x

            x = layers_dict[self.LM](
                (x, decoder_cond), 
                training=training
            ) if self.LM in layers_dict else x

            x = layers_dict[self.DS](
                (x, decoder_cond), 
                training=training
            ) if self.DS in layers_dict else x

            x = layers_dict[self.US](
                (x, decoder_cond), 
                training=training
            ) if self.US in layers_dict else x

            decoder_features_list.append(x)

        x = x[:, 1:] if self.cls_token_type is not None else x

        return x, decoder_cond, decoder_features_list
