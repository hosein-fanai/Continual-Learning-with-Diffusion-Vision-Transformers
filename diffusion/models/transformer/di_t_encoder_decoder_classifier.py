"""Compatibility type combining encoder/decoder and classifier declarations."""

from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.transformer.di_t_encoder_decoder import DiTEncoderDecoder


class DiTEncoderDecoderClassifier(DiTEncoderDecoder, DiTClassifier):
    """Experimental multiple-inheritance encoder/decoder classifier marker.

    The class adds no methods or state.  Python's method-resolution order uses
    ``DiTEncoderDecoder`` first, so construction and calls currently follow that
    class and do not automatically initialize ``DiTClassifier``'s classifier
    branch.  It should therefore be treated as a compatibility/extension point,
    not as a ready-to-train classifier, unless a downstream subclass supplies
    the missing integration.
    """

    pass
