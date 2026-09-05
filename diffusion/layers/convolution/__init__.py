"""Reusable channels-last convolution layers for diffusion networks.

Public convolution layers are imported and cached only when accessed through this
package. Exports cover residual stacks, image resizing, tracked stage containers,
and optional variational flatten/unflatten models; implementation modules define
complete constructor and tensor contracts.
"""

from importlib import import_module


__all__ = (
    "ResidualConvBlock", 
    "ResidualConvStack", 
    "ImageDownsample", 
    "ImageUpsample", 
    "LayerDict", 
    "VariationalReshaper"
)

_LAZY_EXPORTS = {
    "ImageDownsample": (
        "diffusion.layers.convolution.downsample", 
        "ImageDownsample"
    ), 
    "ImageUpsample": (
        "diffusion.layers.convolution.upsample", 
        "ImageUpsample"
    ), 
    "LayerDict": ("diffusion.layers.convolution.stage", "LayerDict"), 
    "ResidualConvBlock": (
        "diffusion.layers.convolution.residual_block", 
        "ResidualConvBlock"
    ), 
    "ResidualConvStack": (
        "diffusion.layers.convolution.residual_block", 
        "ResidualConvStack"
    ), 
    "VariationalReshaper": (
        "diffusion.layers.convolution.variational_reshaper", 
        "VariationalReshaper"
    )
}


def __getattr__(name: str) -> object:
    """Load and cache one public convolution layer on first access.

    Args:
        name (str): Public export name.

    Returns:
        object: Requested layer class or shared layer mapping alias.

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
