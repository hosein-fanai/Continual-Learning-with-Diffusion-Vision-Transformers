"""Lazy public API for diffusion networks, wrappers, layers, and schedules.

Classes from `diffusion.models.transformer` and `diffusion.models.convolution` 
are raw Keras network architectures: they map noisy images, diffusion times, 
and optional conditions to predictions. Classes from `diffusion.models.wrapper` 
own those raw networks and add noise generation, optimization, EMA weights, 
classifier-free guidance, reverse diffusion, evaluation, and Keras 
``fit``/``predict`` hooks. The wrapper is therefore normally the object 
compiled and trained.

This package re-exports the supported high-level models plus reusable blocks,
embeddings, manipulation layers, callbacks, metrics, and the NumPy scheduling
interface. Public objects are imported and cached on first access. Lightweight
Keras registry proxies resolve serializable objects to their canonical classes
during package-only model loading without importing implementation modules or
executing a target module twice under ``python -m``. Constructors and tensor
contracts are documented in their defining modules and package README files.
"""

from importlib import import_module

from common.keras_registry import register_lazy_keras_serializable


_LAZY_EXPORTS = {
    "AdaLNZero": (
        "diffusion.layers.adaptive_layer_normalization_zero", 
        "AdaLNZero"
    ), 
    "BatchLossPlateau": (
        "diffusion.callbacks.batch_loss_plateau", 
        "BatchLossPlateau"
    ), 
    "ConditionEmbedding": (
        "diffusion.layers.embedding.condition_embedding", 
        "ConditionEmbedding"
    ), 
    "DiTClassifier": (
        "diffusion.models.transformer.di_t_classifier", 
        "DiTClassifier"
    ), 
    "DiTDecoder": (
        "diffusion.models.transformer.di_t_decoder", 
        "DiTDecoder"
    ), 
    "DiTDecoderBlock": (
        "diffusion.layers.block.di_t_decoder_block", 
        "DiTDecoderBlock"
    ), 
    "DiTEncoderDecoder": (
        "diffusion.models.transformer.di_t_encoder_decoder", 
        "DiTEncoderDecoder"
    ), 
    "DiTEncoderDecoderClassifier": (
        "diffusion.models.transformer.di_t_encoder_decoder_classifier", 
        "DiTEncoderDecoderClassifier"
    ), 
    "DiffusionClassifier": (
        "diffusion.models.wrapper.diffusion_classifier", 
        "DiffusionClassifier"
    ), 
    "DiffusionClassifierV2": (
        "diffusion.models.wrapper.diffusion_classifier_v2", 
        "DiffusionClassifierV2"
    ), 
    "DiffusionModel": (
        "diffusion.models.wrapper.diffusion_model", 
        "DiffusionModel"
    ), 
    "DiffusionTransformer": (
        "diffusion.models.transformer.diffusion_transformer", 
        "DiffusionTransformer"
    ), 
    "Downsample": (
        "diffusion.layers.manipulation.downsample", 
        "Downsample"
    ), 
    "DropPath": ("diffusion.layers.drop_path", "DropPath"), 
    "EnsembleAccuracy": (
        "diffusion.metrics.ensemble_accuracy", 
        "EnsembleAccuracy"
    ), 
    "FeatureHandler": (
        "diffusion.layers.feature_handler", 
        "FeatureHandler"
    ), 
    "ImageDownsample": (
        "diffusion.layers.convolution", 
        "ImageDownsample"
    ), 
    "ImageGeneratorCallback": (
        "diffusion.callbacks.image_generator_callback", 
        "ImageGeneratorCallback"
    ), 
    "ImageUpsample": ("diffusion.layers.convolution", "ImageUpsample"), 
    "LayerDict": ("diffusion.layers.convolution", "LayerDict"), 
    "LocalMixer": (
        "diffusion.layers.manipulation.local_mixer", 
        "LocalMixer"
    ), 
    "PatchEmbedding": (
        "diffusion.layers.embedding.patch_embedding", 
        "PatchEmbedding"
    ), 
    "RawNetworkValidationCallback": (
        "diffusion.callbacks.raw_network_validation_callback", 
        "RawNetworkValidationCallback"
    ), 
    "ResidualConvBlock": (
        "diffusion.layers.convolution", 
        "ResidualConvBlock"
    ), 
    "ResidualConvStack": (
        "diffusion.layers.convolution", 
        "ResidualConvStack"
    ), 
    "SchedulerName": ("diffusion.schedulers", "SchedulerName"), 
    "SingleTokenLayer": (
        "diffusion.layers.single_token_layer", 
        "SingleTokenLayer"
    ), 
    "UNet": ("diffusion.models.convolution.unet", "UNet"), 
    "UNetClassifier": (
        "diffusion.models.convolution.unet_classifier", 
        "UNetClassifier"
    ), 
    "Upsample": ("diffusion.layers.manipulation.upsample", "Upsample"), 
    "VariationalReshaper": (
        "diffusion.layers.convolution", 
        "VariationalReshaper"
    ), 
    "VisionTransformerBlock": (
        "diffusion.layers.block.vision_transformer_block", 
        "VisionTransformerBlock"
    ), 
    "make_schedule": ("diffusion.schedulers", "make_schedule")
}

_KERAS_SERIALIZABLE_EXPORTS = {
    "ImageDownsample": "diffusion.layers.convolution.downsample", 
    "ImageUpsample": "diffusion.layers.convolution.upsample", 
    "LayerDict": "diffusion.layers.convolution.stage", 
    "ResidualConvBlock": "diffusion.layers.convolution.residual_block", 
    "ResidualConvStack": "diffusion.layers.convolution.residual_block", 
    "UNet": "diffusion.models.convolution.unet", 
    "UNetClassifier": "diffusion.models.convolution.unet_classifier", 
    "VariationalReshaper": "diffusion.layers.convolution.variational_reshaper"
}

for _serializable_name, _module_name in \
_KERAS_SERIALIZABLE_EXPORTS.items():
    register_lazy_keras_serializable(
        _module_name, 
        _serializable_name
    )

__all__ = tuple(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    """Load and cache one documented public diffusion object on first access.

    Args:
        name (str): Public export name.

    Returns:
        object: Requested model, layer, callback, metric, or scheduler object.

    Raises:
        AttributeError: If ``name`` is not part of the public package API.
    """

    # Resolve only documented package exports.
    if name not in _LAZY_EXPORTS:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value

    return value


def __dir__() -> list[str]:
    """Return module globals plus lazily available public exports.

    Returns:
        list[str]: Sorted names discoverable on this package.
    """

    return sorted(set(globals()) | set(__all__))
