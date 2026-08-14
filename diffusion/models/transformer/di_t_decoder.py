import tensorflow as tf
from tensorflow.keras import layers

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer


class DiTDecoder(DiffusionTransformer):
    """
    
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
        feature_aggregation_ids_dict: dict[list[int | None]] = {1: (-1,)}, 
        feature_aggregation_kwargs: dict = {}, 
        cross_attention_aggregation_ids_dict: dict[list[int | None]] = {}, 
        cross_attention_aggregation_kwargs: dict = {}, 
        build: bool = True, 
        **kwargs
    ):
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
        inputs, 
        encoder_cond, 
        encoder_features_list, 
        full_return=False, 
        training=None
    ):
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

    def build_model(self, call_model=True):
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

    def get_causal_attention_mask(self, x):
        seq_len = tf.shape(x)[1]
        mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=tf.bool),
            -1,
            0,
        )

        return mask

    def decode(self, inputs, 
               encoder_cond, 
               encoder_features_list, 
               max_depth=-1, 
               training=None):
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
