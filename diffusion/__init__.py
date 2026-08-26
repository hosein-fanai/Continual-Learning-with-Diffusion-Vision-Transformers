"""Public API for diffusion networks, wrappers, layers, schedules, and tools.

Classes from `diffusion.models.transformer` and `diffusion.models.convolution` 
are raw Keras network architectures: they map noisy images, diffusion times, 
and optional conditions to predictions. Classes from `diffusion.models.wrapper` 
own those raw networks and add noise generation, optimization, EMA weights, 
classifier-free guidance, reverse diffusion, evaluation, and Keras 
``fit``/``predict`` hooks. The wrapper is therefore normally the object 
compiled and trained.

This package re-exports the supported high-level models plus reusable blocks,
embeddings, manipulation layers, callbacks, metrics, and the NumPy scheduling
interface.  Their constructors and tensor contracts are documented in their
defining modules and the package README files.
"""

from .models.wrapper.diffusion_model import DiffusionModel
from .models.wrapper.diffusion_classifier import DiffusionClassifier
from .models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2

from .models.transformer.diffusion_transformer import DiffusionTransformer
from .models.transformer.di_t_classifier import DiTClassifier
from .models.transformer.di_t_decoder import DiTDecoder
from .models.transformer.di_t_encoder_decoder import DiTEncoderDecoder
from .models.transformer.di_t_encoder_decoder_classifier import DiTEncoderDecoderClassifier

from .layers.block.di_t_decoder_block import DiTDecoderBlock
from .layers.block.vision_transformer_block import VisionTransformerBlock
from .layers.embedding.patch_embedding import PatchEmbedding
from .layers.embedding.condition_embedding import ConditionEmbedding
from .layers.manipulation.local_mixer import LocalMixer
from .layers.manipulation.downsample import Downsample
from .layers.manipulation.upsample import Upsample
from .layers.adaptive_layer_normalization_zero import AdaLNZero
from .layers.feature_handler import FeatureHandler
from .layers.drop_path import DropPath
from .layers.single_token_layer import SingleTokenLayer

from .models.convolution.unet import UNet
from .models.convolution.unet_classifier import UNetClassifier

from .layers.convolution import ImageDownsample
from .layers.convolution import ImageUpsample
from .layers.convolution import LayerDict
from .layers.convolution import ResidualConvBlock
from .layers.convolution import ResidualConvStack
from .layers.convolution import VariationalReshaper

from .metrics.ensemble_accuracy import EnsembleAccuracy

from .callbacks.image_generator_callback import ImageGeneratorCallback
from .callbacks.batch_loss_plateau import BatchLossPlateau
from .callbacks.raw_network_validation_callback import RawNetworkValidationCallback

from .schedulers import make_schedule, SchedulerName
