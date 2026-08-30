"""Process-wide reproducibility and numeric-policy setup for experiments.

The helpers in this module are intentionally independent of model, dataset,
and continual-learning implementations.  Call runtime configuration before
constructing any of those objects so Keras initializers and policies observe
the selected settings.
"""

from __future__ import annotations

import tensorflow as tf

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import hashlib

import json


__all__ = (
    "effective_seed", 
    "configure_runtime", 
    "derive_seed", 
    "validate_model_dtype_policy"
)


_NUMPY_SEED_LIMIT = 2 ** 32
_DERIVED_SEED_MODULUS = 2 ** 31 - 1


def _validate_seed(seed: int | None, name: str) -> int | None:
    """Validate and normalize one seed accepted by all global RNG backends.

    Args:
        seed (int | None): Optional non-boolean integer in NumPy's accepted
            unsigned 32-bit seed interval.
        name (str): Parameter name included in validation errors.

    Returns:
        int | None: A plain Python integer, or ``None`` unchanged.

    Raises:
        TypeError: If ``seed`` is not ``None`` or an integral value, or if
            ``name`` is not a nonempty string.
        ValueError: If ``seed`` falls outside ``[0, 2**32)``.
    """

    # Keep validation errors attributable to the public seed argument.
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a nonempty string.")

    # Preserve the explicit unseeded mode without coercion.
    if seed is None:
        return None

    # Reject booleans even though bool is an Integral subclass.
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
    ``task`` is validated for a consistent API but does not change that value.

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
        TypeError: If config sections are missing or inputs have invalid types.
        ValueError: If the selected seed is outside the supported interval.
    """

    # Validate direct task metadata even though one seed serves every task.
    if task is not None and not isinstance(task, str):
        raise TypeError("task must be a string or None.")

    # Read configured values only when a typed-config-compatible object exists.
    if config is not None:
        # A training section is the minimum required config protocol.
        if not hasattr(config, "training"):
            raise TypeError("config must expose a training section.")

        training = config.training
        configured_task = getattr(training, "task", None)

        # Require the configured task field to remain textual when present.
        if configured_task is not None and not isinstance(configured_task, str):
            raise TypeError("config.training.task must be a string or None.")

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
    """Derive an stable TensorFlow-safe child seed for a named RNG stream.

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
    dtype_policy: str
) -> tf.keras.mixed_precision.Policy:
    """Return a validated floating-point Keras dtype policy.

    Args:
        dtype_policy (str): Keras floating-point policy name.

    Returns:
        tf.keras.mixed_precision.Policy: Validated policy with floating compute
        and variable dtypes.

    Raises:
        TypeError: If ``dtype_policy`` is not a string.
        ValueError: If the policy name is invalid or resolves to non-floating
            compute/variable dtypes.
    """

    # Reject arbitrary objects before constructing a Keras policy.
    if not isinstance(dtype_policy, str):
        raise TypeError("dtype_policy must be a string.")
    # Reject empty names before delegating supported-name checks to Keras.
    if not dtype_policy.strip():
        raise ValueError("dtype_policy must be a nonempty policy name.")
    try:
        policy = tf.keras.mixed_precision.Policy(dtype_policy)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Unsupported Keras dtype policy: {dtype_policy!r}."
        ) from error

    compute_dtype = policy.compute_dtype
    variable_dtype = policy.variable_dtype
    # Reject inference-only or nonnumeric policies without concrete dtypes.
    if compute_dtype is None or variable_dtype is None:
        raise ValueError("dtype_policy must define compute and variable dtypes.")

    # This training project only supports floating model computations/weights.
    if not tf.as_dtype(compute_dtype).is_floating \
    or not tf.as_dtype(variable_dtype).is_floating:
        raise ValueError(
            "dtype_policy compute and variable dtypes must be floating point."
        )

    return policy


def validate_model_dtype_policy(
    model: tf.keras.Model, 
    dtype_policy: str | tf.keras.mixed_precision.Policy | None = None, 
    role: str = "model"
) -> str:
    """Reject a restored or nested model with incompatible layer policies.

    Installing a Keras global policy does not rewrite policies already stored
    on deserialized layers. Mixed-float16 models may deliberately retain a
    float32 Dense softmax output for stable probabilities; InputLayers may also
    expose the policy's compute dtype. Every other nested layer must carry the
    requested policy exactly.

    Args:
        model (tf.keras.Model): Restored or externally supplied Keras model.
        dtype_policy (str | tf.keras.mixed_precision.Policy | None): Requested
            policy. None uses the active global policy.
        role (str): Human-readable model role included in validation errors.

    Returns:
        str: Validated requested policy name.

    Raises:
        TypeError: If model, policy, or role has an invalid type.
        ValueError: If any nested layer retains an incompatible policy.
    """

    # Restrict validation to actual Keras models with inspectable sublayers.
    if not isinstance(model, tf.keras.Model):
        raise TypeError("model must be a tf.keras.Model.")
    # Keep error attribution readable for load and teacher call sites.
    if not isinstance(role, str) or not role.strip():
        raise TypeError("role must be a nonempty string.")

    # Resolve an omitted request through the already-installed global policy.
    if dtype_policy is None:
        requested = tf.keras.mixed_precision.global_policy()
    # Preserve a caller-supplied policy instance without string round-tripping.
    elif isinstance(dtype_policy, tf.keras.mixed_precision.Policy):
        requested = dtype_policy
    # Validate a textual policy through the same floating-point restrictions.
    elif isinstance(dtype_policy, str):
        requested = _validate_dtype_policy(dtype_policy)
    # Reject arbitrary dtype objects that lack complete policy semantics.
    else:
        raise TypeError("dtype_policy must be a policy, string, or None.")

    incompatible: list[str] = []
    pending = [model]
    visited: set[int] = set()
    while pending:
        layer = pending.pop()
        # Avoid revisiting shared nested models or layers.
        if id(layer) in visited:
            continue
        visited.add(id(layer))

        layer_policy = getattr(layer, "dtype_policy", None)
        layer_policy_name = getattr(layer_policy, "name", None)
        is_compatible = layer_policy_name == requested.name

        # Functional inputs may explicitly expose the compute dtype itself.
        if isinstance(layer, tf.keras.layers.InputLayer):
            is_compatible = layer_policy_name in {
                requested.name,
                requested.compute_dtype,
                requested.variable_dtype,
            }
        activation = getattr(layer, "activation", None)
        activation_name = getattr(activation, "__name__", None)
        stable_softmax = (
            isinstance(layer, tf.keras.layers.Dense)
            and activation_name == "softmax"
            and layer_policy_name == requested.variable_dtype
        )
        # Permit an explicitly stable Dense probability output under mixed use.
        if stable_softmax:
            is_compatible = True

        # Record every incompatible layer with enough provenance to diagnose it.
        if not is_compatible:
            incompatible.append(
                f"{getattr(layer, 'name', layer.__class__.__name__)}="
                f"{layer_policy_name!r}"
            )
        pending.extend(getattr(layer, "layers", ()))

    # Fail explicitly because changing the global policy cannot repair layers.
    if incompatible:
        preview = ", ".join(incompatible[:8])
        remainder = len(incompatible) - 8
        # Summarize large architectures without hiding the mismatch count.
        if remainder > 0:
            preview += f", ... (+{remainder} more)"
        raise ValueError(
            f"{role} is incompatible with dtype policy {requested.name!r}: "
            f"{preview}. Rebuild or resave it under the requested policy; "
            "setting the global policy does not retrofit serialized layers."
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
        RuntimeError: If this TensorFlow build cannot enable deterministic ops.
    """

    # Require an actual boolean instead of accepting truthy strings or ints.
    if not isinstance(deterministic_ops, bool):
        raise TypeError("deterministic_ops must be a bool.")

    normalized_seed = _validate_seed(seed, "seed")
    policy = _validate_dtype_policy(dtype_policy)

    # TensorFlow deterministic random operations require a configured seed.
    if deterministic_ops and normalized_seed is None:
        raise ValueError(
            "deterministic_ops=True requires an effective random seed."
        )
    # Enable deterministic kernels before any experimental operations execute.
    if deterministic_ops:
        enable_determinism = getattr(
            tf.config.experimental, 
            "enable_op_determinism", 
            None
        )

        # Fail explicitly rather than silently claiming unsupported determinism.
        if not callable(enable_determinism):
            raise RuntimeError(
                "This TensorFlow build cannot enable deterministic operations."
            )

        enable_determinism()

    tf.keras.mixed_precision.set_global_policy(policy)
    # Keras seeds Python, NumPy, and TensorFlow in one version-aware operation.
    if normalized_seed is not None:
        tf.keras.utils.set_random_seed(normalized_seed)

    return policy.name
