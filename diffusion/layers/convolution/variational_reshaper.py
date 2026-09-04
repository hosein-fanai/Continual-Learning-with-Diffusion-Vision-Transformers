"""Functional flatten/unflatten models with an optional variational latent."""

import tensorflow as tf
from tensorflow.keras import layers
from keras.engine.functional import Functional

import math

from typing import Any

from common.argument_saver import ArgumentSaverModel
from common.keras_registry import register_canonical_keras_serializable
from common.runtime import derive_seed

from autoencoder.variational_autoencoder import VariationalAutoencoder


def _sample_latent(
    values: tuple[tf.Tensor, tf.Tensor],
    seed: int | None = None,
    dtype: tf.dtypes.DType | str | None = None,
) -> tf.Tensor:
    """Sample a reparameterized latent vector from Gaussian parameters.

    Args:
        values (tuple[tf.Tensor, tf.Tensor]): Latent mean and log-variance
            tensors with identical shapes.
        seed (int | None): Optional TensorFlow operation seed.
        dtype (tf.dtypes.DType | str | None): Stable calculation dtype.

    Returns:
        tf.Tensor: Reparameterized sample with the same shape and dtype.
    """

    return VariationalAutoencoder.compute_z(
        values[0],
        values[1],
        seed=seed,
        dtype=dtype,
    )


def _batch_size(value: tf.Tensor) -> tf.Tensor:
    """Return the dynamic batch size used for deterministic dummy outputs.

    Args:
        value (tf.Tensor): Batched input tensor.

    Returns:
        tf.Tensor: Scalar integer batch size.
    """

    return tf.shape(value)[0]


@register_canonical_keras_serializable(package="continual_learning")
class VariationalReshaper(ArgumentSaverModel, Functional):
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
        seed: int | None = None,
        **kwargs: Any
    ) -> None:
        """Build a functional Keras model with statically known reshape sizes.

        Args:
            reshape_type (str): Either ``"flatten"`` or ``"unflatten"``.
            source_shape (tuple[int, ...] | list[int]): Positive static image
                feature dimensions, excluding batch.
            add_kl (bool): Whether flattening creates Gaussian latent parameters.
            latent_dim_ratio (float): Positive latent-to-flattened-width ratio.
            seed (int | None): Parent seed used to derive this bottleneck's
                reparameterization stream.
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
        reparameterization_seed = derive_seed(
            seed,
            "variational_reshaper",
            model_name,
            "reparameterization",
        )
        dtype = kwargs.pop("dtype", None)
        kwargs.pop("dynamic", None)
        # Resolve omitted dtypes through the active global numeric policy.
        if dtype is None:
            policy = tf.keras.mixed_precision.global_policy()
        # Preserve an already-resolved mixed-precision policy.
        elif isinstance(dtype, tf.keras.mixed_precision.Policy):
            policy = dtype
        # Convert an explicit dtype name into a uniform Keras policy.
        else:
            policy = tf.keras.mixed_precision.Policy(dtype)
        input_dtype = policy.compute_dtype
        layer_dtype_kwargs = {"dtype": policy}

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
                arguments={
                    "seed": reparameterization_seed,
                    "dtype": policy.variable_dtype,
                },
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
        self._save_init_args({
            "reshape_type": reshape_type, 
            "source_shape": source_shape, 
            "add_kl": add_kl, 
            "latent_dim_ratio": latent_dim_ratio,
            "seed": seed,
        })
        self._init_config.update(layers.Layer.get_config(self))
        self.source_shape_ = source_shape
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
        """Validate the reshape direction and preserve a usable KL width.

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
        # Reject KL ratios that would round the latent width down to zero.
        if add_kl and reshape_type == "flatten" \
        and int(math.prod(source_shape) * latent_dim_ratio) < 1:
            raise ValueError("latent_dim_ratio creates an empty latent vector.")

    def get_config(self) -> dict[str, Any]:
        """Return ArgumentSaver state without the generated functional graph.

        Args:
            None.

        Returns:
            dict[str, Any]: JSON-compatible constructor configuration.
        """

        config = super().get_config()
        # Exclude the Functional graph so inherited from_config receives only
        # VariationalReshaper constructor arguments under TensorFlow 2.10.
        for key in ("layers", "input_layers", "output_layers"):
            config.pop(key, None)

        return config


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
        seed=37,
        name="depth_2_reshaper"
    )
    z, mean, log_var = variational(x)
    assert z.shape == (2, 48)
    assert mean.shape == log_var.shape == (2, 24)
    assert variational.output_shape[1][-1] == 24
    assert variational.get_layer("depth_2_reshaper/z") is not None
    assert variational.get_config()["seed"] == 37

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
    assert clone.seed == 37

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
