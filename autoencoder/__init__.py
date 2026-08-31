"""Lazy autoencoder API with canonical Keras deserialization registration."""

from importlib import import_module

from common.keras_registry import register_lazy_keras_serializable


__all__ = (
    "DecoderAccuracyCallback", 
    "VAEClassifier", 
    "VariationalAutoencoder"
)

_LAZY_EXPORTS = {
    "DecoderAccuracyCallback": (
        "autoencoder.decoder_accuracy_callback", 
        "DecoderAccuracyCallback"
    ), 
    "VAEClassifier": ("autoencoder.vae_classifier", "VAEClassifier"), 
    "VariationalAutoencoder": (
        "autoencoder.variational_autoencoder", 
        "VariationalAutoencoder"
    )
}

for _serializable_name in ("VAEClassifier", "VariationalAutoencoder"):
    _module_name, _attribute_name = _LAZY_EXPORTS[_serializable_name]
    register_lazy_keras_serializable(
        _module_name, 
        _attribute_name, 
        aliases=(_serializable_name,)
    )


def __getattr__(name: str) -> object:
    """Load and cache one public autoencoder object on first access.

    Args:
        name (str): Public export name.

    Returns:
        object: Requested model or callback class.

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
