from common.argument_saver import ArgumentSaverModel

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.transformer.di_t_decoder import DiTDecoder


class DiTEncoderDecoder(ArgumentSaverModel):

    def __init__(
        self, 
        encoder_kwargs={}, 
        decoder_kwargs={}, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._save_init_args(locals())

        self.encoder = DiffusionTransformer(
            **self.encoder_kwargs, 
            name_prefix="encoder_model/"
        )
        self.decoder = DiTDecoder(
            **self.decoder_kwargs, 
            name_prefix="decoder_model/"
        )

        self.image_size = self.encoder.image_size
        self.channels = self.encoder.channels
        self.use_cfg = self.encoder.use_cfg
        self.timesteps = self.encoder.timesteps

        self.supports_teacher_forcing = True

        self.build(())

    def call(
        self, 
        inputs, 
        full_return=False, 
        training=None
    ):
        noisy_images, ts, labels, decoder_input_images = inputs

        _, encoder_cond, encoder_features_list = self.encoder.encode(
            (noisy_images, ts, labels), 
            training=training
        )

        outputs = self.decoder(
            (decoder_input_images, ts, labels), 
            encoder_cond, encoder_features_list, 
            full_return=full_return, 
            training=training,
        )

        return outputs
