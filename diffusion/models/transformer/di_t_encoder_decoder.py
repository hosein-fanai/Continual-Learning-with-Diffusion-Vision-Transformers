"""Composition model joining a diffusion transformer encoder and DiT decoder."""

import tensorflow as tf

from common.argument_saver import ArgumentSaverModel

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.transformer.di_t_decoder import DiTDecoder


class DiTEncoderDecoder(ArgumentSaverModel):
    """Compose raw encoder and decoder networks for teacher-forced prediction.

    ``encoder`` is a :class:`DiffusionTransformer` that produces depth-indexed
    features.  ``decoder`` is a :class:`DiTDecoder` intended to consume those
    features plus a shifted decoder image.  This is a raw network composition;
    :class:`diffusion.models.wrapper.diffusion_encoder_decoder_model.DiffusionEncoderDecoderModel`
    supplies its specialized training step.

    Note:
        This path is experimental.  The current :meth:`call` unpacks three
        values from ``DiffusionTransformer.encode``, whose current API returns
        five, and the decoder has legacy structured-input limitations.  The
        class documents the intended interface but callers should validate or
        update the integration before production use.

    Attributes:
        encoder (DiffusionTransformer): Raw encoding/noise-transformer branch.
        decoder (DiTDecoder): Raw teacher-forcing decoder branch.
        supports_teacher_forcing (bool): Always true after initialization.
    """

    def __init__(
        self, 
        encoder_kwargs: dict[str, object] = {},
        decoder_kwargs: dict[str, object] = {},
        **kwargs
    ):
        """Initialize the encoder/decoder pair.

        Args:
            encoder_kwargs (dict[str, object]): Keyword arguments accepted by
                ``DiffusionTransformer``.  Common keys include ``image_size``,
                ``channels``, ``patch_size``, ``dim``, ``depth``, all
                ``*_ids_dict`` routing mappings, component ``*_kwargs`` option
                dictionaries, conditioning/token options, and ``build``.  The
                composition supplies ``name_prefix="encoder_model/"``.
            decoder_kwargs (dict[str, object]): Keyword arguments accepted by
                ``DiTDecoder``.  It normally must include
                ``encoder_output_grid_size`` (int) and ``encoder_output_dim``
                (int), and may include ``use_causal_mask``,
                ``decoder_separate_cond``, encoder aggregation mappings/options,
                and every ``DiffusionTransformer`` option.  The composition
                supplies ``name_prefix="decoder_model/"``.
            **kwargs: Standard ``tf.keras.Model`` constructor keys: ``name``
                (str), ``trainable`` (bool), ``dtype`` (dtype name/policy), and
                ``dynamic`` (bool).

        Returns:
            None.  Submodels are created, public image/schedule attributes are
            copied from the encoder, and the composition is marked built.
        """
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
        inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        full_return: bool = False,
        training: bool | None = None
    ) -> dict[str, object]:
        """Encode an image and decode a teacher-forcing image.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]): Encoder
                noisy images ``[B,H,W,C]``, timestep IDs ``[B]``, label IDs
                ``[B]``, and decoder/teacher-forcing images ``[B,H,W,C]``.
            full_return (bool): Request encoder/decoder intermediates from the
                decoder result.
            training (bool | None): Keras training mode for both submodels.

        Returns:
            dict[str, object]: Decoder result containing at least ``"noises"``;
            when ``full_return=True`` it also contains encoder and decoder
            conditions/features as described by ``DiTDecoder.call``.
        """
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
