"""Functional flatten/unflatten models with an optional variational latent."""

import tensorflow as tf
from tensorflow.keras import layers, models

import math

from typing import Any

from autoencoder.variational_autoencoder import VariationalAutoencoder


def _sample_latent(values: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Sample a reparameterized latent vector from Gaussian parameters.

    Args:
        values (tuple[tf.Tensor, tf.Tensor]): Latent mean and log-variance
            tensors with identical shapes.

    Returns:
        tf.Tensor: Reparameterized sample with the same shape and dtype.
    """

    return VariationalAutoencoder.compute_z(values[0], values[1])


def _batch_size(value: tf.Tensor) -> tf.Tensor:
    """Return the dynamic batch size used for deterministic dummy outputs.

    Args:
        value (tf.Tensor): Batched input tensor.

    Returns:
        tf.Tensor: Scalar integer batch size.
    """

    return tf.shape(value)[0]


@tf.keras.utils.register_keras_serializable(package="continual_learning")
class VariationalReshaper(models.Model):
    """Flatten or restore one static image-feature shape.

    The model always returns ``(x, mean, log_variance)``. A KL-enabled flatten
    samples a latent and projects it back to the flattened width when
    ``latent_dim_ratio != 1``. The projection is named ``<model-name>/z`` so
    ``DiffusionModel.sample_vae`` can retrieve it without wrapper changes.
    """

    def __init__(
        self, 
        reshape_type: str, 
        source_shape: tuple[int, ...] | list[int], 
        add_kl: bool = False, 
        latent_dim_ratio: float = 1.0, 
        **kwargs: Any
    ) -> None:
        """Build a functional Keras model with statically known reshape sizes.

        Args:
            reshape_type (str): Either ``"flatten"`` or ``"unflatten"``.
            source_shape (tuple[int, ...] | list[int]): Positive static image
                feature dimensions, excluding batch.
            add_kl (bool): Whether flattening creates Gaussian latent parameters.
            latent_dim_ratio (float): Positive latent-to-flattened-width ratio.
            **kwargs (Any): Standard Keras model options.

        Returns:
            None: Initialization constructs the functional model graph.
        """

        source_shape = tuple(source_shape)
        self._check_arguments(
            reshape_type, 
            source_shape, 
            add_kl, 
            latent_dim_ratio, 
        )
        kwargs = dict(kwargs)
        model_name = kwargs.get("name", None) or "variational_reshaper"
        dtype = kwargs.pop("dtype", None)
        kwargs.pop("dynamic", None)
        # Use a mixed-precision policy's compute dtype for the functional input.
        if isinstance(dtype, tf.keras.mixed_precision.Policy):
            input_dtype = dtype.compute_dtype
        # Otherwise use the explicit dtype or TensorFlow's float32 default.
        else:
            input_dtype = dtype or tf.float32
        layer_dtype_kwargs = {} if dtype is None else {"dtype": dtype}

        flattened_dim = math.prod(source_shape)
        # Build an image-shaped input and flattening path for encoder use.
        if reshape_type == "flatten":
            inputs = layers.Input(
                shape=source_shape, 
                dtype=input_dtype, 
                name=f"{model_name}/inputs"
            )
            x = layers.Flatten(
                name=f"{model_name}/flatten", 
                **layer_dtype_kwargs,
            )(inputs)
        # Build a vector input and unflattening path for decoder use.
        else:
            inputs = layers.Input(
                shape=(flattened_dim,), 
                dtype=input_dtype, 
                name=f"{model_name}/inputs"
            )
            x = layers.Reshape(
                source_shape, 
                name=f"{model_name}/unflatten", 
                **layer_dtype_kwargs
            )(inputs)

        latent_dim = None
        # Add Gaussian latent sampling only to KL-enabled flattening paths.
        if add_kl and reshape_type == "flatten":
            latent_dim = int(flattened_dim * latent_dim_ratio)
            z_mean = layers.Dense(
                latent_dim, 
                name=f"{model_name}/z_mean", 
                **layer_dtype_kwargs
            )(x)
            z_log_var = layers.Dense(
                latent_dim, 
                name=f"{model_name}/z_log_var", 
                **layer_dtype_kwargs
            )(x)
            z = layers.Lambda(
                _sample_latent,
                name=f"{model_name}/sample"
            )((z_mean, z_log_var))
            x = layers.Dense(
                flattened_dim, 
                name=f"{model_name}/z", 
                **layer_dtype_kwargs
            )(z) if latent_dim_ratio != 1.0 else z
        # Return batch-sized dummy statistics when no variational latent exists.
        else:
            dummy = layers.Lambda(
                _batch_size,
                name=f"{model_name}/dummy"
            )(inputs)
            z_mean, z_log_var = dummy, dummy

        super().__init__(
            inputs=inputs, 
            outputs=(x, z_mean, z_log_var), 
            **kwargs
        )
        self.reshape_type = reshape_type
        self.source_shape_ = source_shape
        self.add_kl = add_kl
        self.latent_dim_ratio = latent_dim_ratio
        self.flattened_dim = flattened_dim
        self.latent_dim = latent_dim
        self.output_dim = flattened_dim if reshape_type == "flatten" \
            else source_shape[-1]

    @staticmethod
    def _check_arguments(
        reshape_type: str, 
        source_shape: tuple[int, ...], 
        add_kl: bool, 
        latent_dim_ratio: float
    ) -> None:
        """Validate static dimensions and the supported reshape direction.

        Args:
            reshape_type (str): Candidate reshape direction.
            source_shape (tuple[int, ...]): Candidate positive static shape.
            add_kl (bool): Candidate variational-mode flag.
            latent_dim_ratio (float): Candidate positive latent-width ratio.

        Returns:
            None: Valid arguments complete without a value.
        """

        # Restrict reshaping to the two supported directions.
        if reshape_type not in ("flatten", "unflatten"):
            raise ValueError("reshape_type must be flatten or unflatten.")
        # Require a non-empty source shape containing positive integers.
        if len(source_shape) == 0 or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 1
            for dim in source_shape
        ):
            raise ValueError("source_shape must contain positive integers.")
        # Require an explicit boolean variational-mode flag.
        if not isinstance(add_kl, bool):
            raise ValueError("add_kl must be boolean.")
        # Require a finite positive numeric latent-width ratio.
        if not isinstance(latent_dim_ratio, (int, float)) \
        or isinstance(latent_dim_ratio, bool) \
        or not math.isfinite(latent_dim_ratio) \
        or latent_dim_ratio <= 0.0:
            raise ValueError("latent_dim_ratio must be finite and positive.")
        # Reject KL ratios that would round the latent width down to zero.
        if add_kl and reshape_type == "flatten" \
        and int(math.prod(source_shape) * latent_dim_ratio) < 1:
            raise ValueError("latent_dim_ratio creates an empty latent vector.")

    def get_config(self) -> dict[str, Any]:
        """Return only constructor state, not the generated functional graph.

        Args:
            None.

        Returns:
            dict[str, Any]: JSON-compatible constructor configuration.
        """

        config = layers.Layer.get_config(self)
        config.update({
            "reshape_type": self.reshape_type, 
            "source_shape": list(self.source_shape_), 
            "add_kl": self.add_kl, 
            "latent_dim_ratio": self.latent_dim_ratio, 
        })

        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VariationalReshaper":
        """Rebuild the functional graph from the compact constructor config.

        Args:
            config (dict[str, Any]): Serialized constructor configuration.

        Returns:
            VariationalReshaper: Reconstructed functional model.
        """

        return cls(**dict(config))


def run_self_tests() -> dict[str, str]:
    """Test deterministic, KL, projected-latent, and unflatten branches.

    Args:
        None.

    Returns:
        dict[str, str]: One success entry after all checks pass.
    """

    tf.random.set_seed(103)
    x = tf.random.normal((2, 4, 4, 3))

    plain = VariationalReshaper(
        "flatten", 
        (4, 4, 3), 
        name="depth_1_reshaper"
    )
    flat, mean, log_var = plain(x)
    assert flat.shape == (2, 48)
    assert mean.shape == log_var.shape == tf.TensorShape([])

    variational = VariationalReshaper(
        "flatten", 
        (4, 4, 3), 
        add_kl=True, 
        latent_dim_ratio=0.5, 
        name="depth_2_reshaper"
    )
    z, mean, log_var = variational(x)
    assert z.shape == (2, 48)
    assert mean.shape == log_var.shape == (2, 24)
    assert variational.output_shape[1][-1] == 24
    assert variational.get_layer("depth_2_reshaper/z") is not None

    unflatten = VariationalReshaper(
        "unflatten", 
        (4, 4, 3), 
        add_kl=True, 
        name="depth_3_reshaper"
    )
    restored, _, _ = unflatten(flat)
    assert restored.shape == x.shape

    clone = VariationalReshaper.from_config(variational.get_config())
    assert clone(x)[0].shape == z.shape
    assert clone.output_shape[1][-1] == 24

    try:
        VariationalReshaper("flatten", (1,), add_kl=True, latent_dim_ratio=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("An empty latent width must fail.")

    return {"VariationalReshaper": "passed"}


# Run the module's focused self-tests when executed directly.
if __name__ == "__main__":
    print(run_self_tests())
