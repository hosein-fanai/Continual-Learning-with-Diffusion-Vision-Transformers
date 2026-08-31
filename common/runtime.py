"""Process-wide reproducibility and numeric-policy setup for experiments.

The helpers in this module are intentionally independent of model, dataset,
and continual-learning implementations.  Call runtime configuration before
constructing any of those objects so Keras initializers and policies observe
the selected settings.
"""

from __future__ import annotations

import tensorflow as tf

import hashlib

import json

from collections.abc import Sequence
from numbers import Integral
from typing import Any


__all__ = (
    "effective_seed", 
    "configure_runtime", 
    "derive_seed", 
    "validate_model_dtype_policy"
)

_NUMPY_SEED_LIMIT = 2 ** 32
_DERIVED_SEED_MODULUS = 2 ** 31 - 1


def _validate_seed(seed: int | None, name: str) -> int | None:
    """Normalize one seed accepted by all global RNG backends.

    Args:
        seed (int | None): Optional NumPy-compatible seed.
        name (str): Parameter name included in validation errors.

    Returns:
        int | None: A plain Python integer, or ``None`` unchanged.

    Raises:
        TypeError: If ``seed`` is not a non-boolean integer or ``None``.
        ValueError: If ``seed`` falls outside NumPy's supported interval.
    """

    # Preserve the explicit unseeded mode without coercion.
    if seed is None:
        return None

    # Reject values that merely coerce to an integer, including booleans.
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError(f"{name} must be a non-boolean integer or None.")
    normalized = int(seed)
    # Keras seeds NumPy as well as Python and TensorFlow, so use their overlap.
    if not 0 <= normalized < _NUMPY_SEED_LIMIT:
        raise ValueError(f"{name} must be in [0, 2**32).")

    return normalized


def effective_seed(
    config: Any | None = None, 
    seed: int | None = None, 
    task: str | None = None
) -> int | None:
    """Resolve the authoritative seed for configured or direct execution.

    A configured continual run uses ``continually_learn.seed`` when present
    and otherwise falls back to ``training.seed``.  Configured non-continual
    runs use only ``training.seed``.  Direct execution has one ``seed`` input;
    ``task`` remains compatibility metadata and does not change that value.

    Args:
        config (Any | None): Optional repository config tree exposing a
            ``training`` section and, for continual overrides, an optional
            ``continually_learn`` section.
        seed (int | None): Direct-mode seed used only when ``config`` is None.
        task (str | None): Optional direct-mode task name.

    Returns:
        int | None: The validated effective seed, or ``None`` for an unseeded
        run.

    Raises:
        TypeError: If the selected seed has an invalid type.
        ValueError: If the selected seed is outside the supported interval.
    """

    # Read configured values only when a typed-config-compatible object exists.
    if config is not None:
        training = config.training
        configured_task = getattr(training, "task", None)

        selected_seed = getattr(training, "seed", None)
        is_continual = str(configured_task).lower() == "continual"

        # Give the continual section authority only for continual experiments.
        if is_continual:
            continual_config = getattr(config, "continually_learn", None)
            continual_seed = getattr(continual_config, "seed", None)
            # Fall back to training.seed when the continual override is absent.
            if continual_seed is not None:
                selected_seed = continual_seed

        return _validate_seed(selected_seed, "effective seed")

    return _validate_seed(seed, "seed")


def derive_seed(
    seed: int | None, 
    *parts: str | int
) -> int | None:
    """Derive a stable TensorFlow-safe child seed for a named RNG stream.

    The derivation uses canonical JSON and SHA-256 rather than Python's salted
    ``hash`` function, so it is stable across processes and platforms.  A
    caller can use components such as ``("dataloader", task_index, epoch)``
    to isolate streams without making their results depend on call order.

    Args:
        seed (int | None): Valid experiment seed; ``None`` preserves
            unseeded behavior.
        *parts (str | int): One or more stable stream identifiers.

    Returns:
        int | None: A deterministic integer in ``[0, 2**31 - 1)``, or ``None``
        when ``seed`` is ``None``.

    Raises:
        TypeError: If a component is a boolean or is not a string/integer.
        ValueError: If no component is supplied or the master seed is outside
            the supported interval.
    """

    normalized_master = _validate_seed(seed, "seed")

    # Every child stream must have at least one explicit identity component.
    if not parts:
        raise ValueError("At least one seed component is required.")

    normalized_components: list[tuple[str, str | int]] = []
    for component in parts:
        # Tag strings and integers so values such as 1 and "1" cannot collide.
        if isinstance(component, str):
            normalized_components.append(("str", component))
        # Accept NumPy integer scalars while continuing to reject booleans.
        elif isinstance(component, Integral) and not isinstance(component, bool):
            normalized_components.append(("int", int(component)))
        # Reject unstable representations of arbitrary Python objects.
        else:
            raise TypeError(
                "Seed components must be non-boolean integers or strings."
            )

    # Preserve unseeded behavior after validating the stream description.
    if normalized_master is None:
        return None

    payload: Sequence[object] = (normalized_master, normalized_components)
    encoded = json.dumps(
        payload, 
        ensure_ascii=False, 
        separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()

    return int.from_bytes(digest[:8], "big") % _DERIVED_SEED_MODULUS


def _validate_dtype_policy(
    dtype_policy: str | tf.keras.mixed_precision.Policy
) -> tf.keras.mixed_precision.Policy:
    """Resolve a Keras policy and require floating-point computation.

    Args:
        dtype_policy (str | tf.keras.mixed_precision.Policy): Keras policy.

    Returns:
        tf.keras.mixed_precision.Policy: Keras policy object.
    """

    # Reuse complete policy objects without string round-tripping.
    if isinstance(dtype_policy, tf.keras.mixed_precision.Policy):
        policy = dtype_policy
    # Let Keras interpret names and report malformed policy inputs.
    else:
        policy = tf.keras.mixed_precision.Policy(dtype_policy)

    dtypes = (policy.compute_dtype, policy.variable_dtype)
    # Exclude integer-only policies that cannot support scientific training.
    if not all(dtype is not None and tf.as_dtype(dtype).is_floating
               for dtype in dtypes):
        raise ValueError(
            "dtype_policy must use floating compute and variable dtypes."
        )
    return policy


def validate_model_dtype_policy(
    model: tf.keras.Model, 
    dtype_policy: str | tf.keras.mixed_precision.Policy | None = None, 
    role: str = "model"
) -> str:
    """Check policies retained by a restored model and its layers.

    Args:
        model (tf.keras.Model): Restored or externally supplied Keras model.
        dtype_policy (str | tf.keras.mixed_precision.Policy | None): Requested
            policy. None uses the active global policy.
        role (str): Model role included in a mismatch error.

    Returns:
        str: Validated requested policy name.

    Raises:
        ValueError: If the model or a nested layer has an incompatible policy.
    """

    # Resolve an omitted request through the already-installed global policy.
    requested = _validate_dtype_policy(
        tf.keras.mixed_precision.global_policy()
        if dtype_policy is None else dtype_policy
    )

    incompatible: list[str] = []
    layers = {id(layer): layer for layer in (model, *model.submodules)}.values()
    for layer in layers:
        policy_name = getattr(
            getattr(layer, "dtype_policy", None), "name", None
        )
        allowed = {requested.name}
        # Functional inputs may expose either dtype of a mixed policy.
        if isinstance(layer, tf.keras.layers.InputLayer):
            allowed.update((requested.compute_dtype, requested.variable_dtype))
        activation = getattr(getattr(layer, "activation", None), "__name__", None)
        # Permit explicitly stable probability outputs at variable precision.
        if isinstance(layer, tf.keras.layers.Dense) and activation == "softmax":
            allowed.add(requested.variable_dtype)
        # Keep compact provenance for every incompatible nested layer.
        if policy_name not in allowed:
            incompatible.append(f"{layer.name}={policy_name!r}")

    # Fail once with the collected layer context needed to rebuild the model.
    if incompatible:
        raise ValueError(
            f"{role} has layers incompatible with dtype policy "
            f"{requested.name!r}: {', '.join(incompatible[:8])}. "
            "Global policy changes do not retrofit restored layers."
        )

    return requested.name


def configure_runtime(
    seed: int | None = None, 
    dtype_policy: str = "float32", 
    deterministic_ops: bool = False
) -> str:
    """Install process-wide RNG, dtype, and deterministic-operation settings.

    This function must run before dataset or model construction.  Setting
    ``deterministic_ops=False`` leaves TensorFlow's current process-wide
    determinism state unchanged because TensorFlow 2.10 does not expose a
    symmetric disable operation.

    Args:
        seed (int | None): Effective experiment seed already resolved by
            :func:`effective_seed`.
        dtype_policy (str): Floating Keras global policy name.
        deterministic_ops (bool): Enable deterministic TensorFlow kernels.

    Returns:
        str: Name of the installed Keras dtype policy.

    Raises:
        TypeError: If seed, policy, or boolean inputs have invalid types.
        ValueError: If a seed/policy is invalid, or deterministic operations
            are requested without an effective seed.
        AttributeError: If TensorFlow cannot enable deterministic operations.
    """

    normalized_seed = _validate_seed(seed, "seed")
    policy = _validate_dtype_policy(dtype_policy)

    # Prevent truthy strings or numbers from changing process-wide behavior.
    if not isinstance(deterministic_ops, bool):
        raise TypeError("deterministic_ops must be a boolean.")
    # TensorFlow deterministic random operations require a configured seed.
    if deterministic_ops and normalized_seed is None:
        raise ValueError(
            "deterministic_ops=True requires an effective random seed."
        )
    # Enable deterministic kernels before any experimental operations execute.
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()

    tf.keras.mixed_precision.set_global_policy(policy)
    # Keras seeds Python, NumPy, and TensorFlow in one version-aware operation.
    if normalized_seed is not None:
        tf.keras.utils.set_random_seed(normalized_seed)

    return policy.name
