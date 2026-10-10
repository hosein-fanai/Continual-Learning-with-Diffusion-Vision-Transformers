"""Opt-in conditional search for the native convolutional epsilon denoiser.

The denoiser_v1 space is separate from the legacy U-Net choices. Its marker and
all overrides are sealed by the shared HPO study specification. Width templates
span two to four encoder levels at CIFAR's 32-pixel resolution; every level
halves the spatial side and has a matching decoder skip. Residual kernels remain
3x3 because the public UNet constructor does not expose kernel size, attention,
or GroupNorm. Normalization means the implemented BatchNormalization or none.

Embeddings share a total channel budget: balanced divides it approximately
equally, and time_rich assigns half to time and a quarter each to image/label.
Interpolation is sampled only for resize-based upsamplers; transposed convolution
has no interpolation hyperparameter. No latent reshaper, auxiliary classifier,
progressive stage, or skip ablation changes the ordinary denoising objective.

The notebook recipe fixes the noising process and MSE objective across trials.
Guidance, sampling steps, and eta affect generated images rather than the loss
used for search, so they remain a separate finalist evaluation choice. These
constants are framework-free and can be imported by the remote coordinator.
"""

from __future__ import annotations

from typing import Any


SPACE_NAME = "denoiser_v1"
ARCHITECTURE_CHOICES = {
    "unet_architecture_space": [SPACE_NAME], 
    "widths": [
        "32-64", "32-64-96", "32-64-128", "48-96-192", 
        "64-96-128", "64-128-256", "32-64-128-256"
    ], 
    "block_depth": [1, 2, 3], 
    "bottleneck_mult": [1.0, 1.5, 2.0], 
    "bottleneck_depth": [1, 2, 3], 
    "embedding_dim": [64, 96, 128, 192], 
    "embedding_layout": ["balanced", "time_rich"], 
    "batch_norm": [False, True], 
    "dropout": [0.0, 0.05, 0.1, 0.2], 
    "activation_func": ["swish", "gelu", "relu"], 
    "downsampling_method": ["avg_pooling", "max_pooling", "cnn_stride"], 
    "upsampling_method": ["interpolate", "cnn_interpolate", "cnn_transpose"], 
    "upsampling_interpolation": ["nearest", "bilinear"]
}
SEARCH_SPACE_OVERRIDES = {
    **ARCHITECTURE_CHOICES, 
    "batch_size": [32, 64, 128], 
    "optimizer": ["adam", "adamw"], 
    "learning_rate": {"low": 1e-4, "high": 1e-3, "log": True}, 
    "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True}, 
    "clipnorm": [None], 
    "global_clipnorm": [None, 1.0, 5.0], 
    "learning_rate_schedule": ["cosine"], 
    "loss_function": ["mse"], 
    "timesteps": [1000], 
    "schedule": ["clipped_cosine"], 
    "ema_decay": [0.995, 0.999], 
    "p_uncond": [0.05, 0.1, 0.2], 
    "image_loss_coef": [0.0]
}


def suggest_denoiser(trial: Any) -> dict[str, object]:
    """Resolve only active native U-Net settings from a wrapped Optuna trial.

    Args:
        trial (Any): Shared HPO trial view, including immutable override handling.

    Returns:
        dict[str, object]: Public UNet constructor values; condition vocabulary,
            channels, image size, and seed remain owned by the shared factory.

    Raises:
        ValueError: The trial's override selects an unsupported categorical value.
    """

    trial.suggest_categorical("unet_architecture_space", [SPACE_NAME])
    widths_name = trial.suggest_categorical("widths", ARCHITECTURE_CHOICES["widths"])
    widths = tuple(int(value) for value in widths_name.split("-"))
    embedding_dim = trial.suggest_categorical("embedding_dim", ARCHITECTURE_CHOICES["embedding_dim"])
    layout = trial.suggest_categorical("embedding_layout", ARCHITECTURE_CHOICES["embedding_layout"])
    # Both allocations preserve the same total projection/conditioning width.
    if layout == "time_rich":
        image_dim = label_dim = embedding_dim // 4
        time_dim = embedding_dim - image_dim - label_dim
    # The balanced layout gives any remainder to the timestep embedding.
    else:
        image_dim, remainder = divmod(embedding_dim, 3)
        label_dim = image_dim
        time_dim = image_dim + remainder
    upsampling = trial.suggest_categorical("upsampling_method", ARCHITECTURE_CHOICES["upsampling_method"])
    # The constructor retains a valid inert default for transposed convolution.
    interpolation = trial.suggest_categorical(
        "upsampling_interpolation", ARCHITECTURE_CHOICES["upsampling_interpolation"]
    ) if upsampling != "cnn_transpose" else "bilinear"
    return {
        "widths": widths, 
        "block_depth": trial.suggest_categorical("block_depth", ARCHITECTURE_CHOICES["block_depth"]), 
        "bottleneck_width": int(max(widths) * trial.suggest_categorical(
            "bottleneck_mult", ARCHITECTURE_CHOICES["bottleneck_mult"]
        )), 
        "bottleneck_depth": trial.suggest_categorical("bottleneck_depth", ARCHITECTURE_CHOICES["bottleneck_depth"]), 
        "image_embedding_dim": image_dim, 
        "time_embedding_dim": time_dim, 
        "label_embedding_dim": label_dim, 
        "use_batch_norm": trial.suggest_categorical("batch_norm", ARCHITECTURE_CHOICES["batch_norm"]), 
        "dropout_rate": trial.suggest_categorical("dropout", ARCHITECTURE_CHOICES["dropout"]), 
        "activation_func": trial.suggest_categorical("activation_func", ARCHITECTURE_CHOICES["activation_func"]), 
        "downsampling_method": trial.suggest_categorical("downsampling_method", ARCHITECTURE_CHOICES["downsampling_method"]), 
        "upsampling_method": upsampling, 
        "upsampling_interpolation": interpolation, 
        "use_skip_connections": True, 
        "final_activation_func": "linear"
    }
