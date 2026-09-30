"""Small Keras 3 bridges shared by project models and diagnostics."""

import tensorflow as tf

from copy import copy, deepcopy

from collections.abc import Sequence


NAME_SEPARATOR = "__"


def compute_compiled_loss(
    model: tf.keras.Model, 
    y_true: tf.Tensor, 
    y_pred: tf.Tensor, 
    sample_weight: tf.Tensor | None = None, 
    regularization_losses: Sequence[tf.Tensor] = ()
) -> tf.Tensor:
    """Evaluate a single-output compiled loss in the model's stable precision.

    Keras 3 creates its loss container and string/function loss wrappers using
    the global floatx setting, independently of the model's dtype policy. Align
    their dtype before evaluation so float64 residuals are never rounded through
    float32. Mixed policies retain float32 loss math. Copy resolved loss objects
    before changing their dtype so another model can share the original loss.

    Args:
        model (tf.keras.Model): Compiled model owning the Keras 3 loss container.
        y_true (tf.Tensor): Target tensor accepted by the compiled loss, such as
            integer sparse class IDs [B] or floating reconstruction targets [B, ...].
        y_pred (tf.Tensor): Prediction tensor, such as class scores [B, K] or
            reconstruction values [B, ...], cast by the loss to stable variable dtype.
        sample_weight (tf.Tensor | None): Numeric scalar or tensor broadcastable
            to per-example/per-element loss values; None means unweighted loss.
            Defaults to ``None``.
        regularization_losses (Sequence[tf.Tensor]): Already evaluated numeric
            layer penalties, usually scalar, cast to variable dtype and added once.
            Defaults to (), evaluating only the data loss.

    Returns:
        loss (tf.Tensor): Native weighted/reduced loss plus supplied penalties,
            in the model's variable dtype (float32 for mixed policies or float64
            for a float64 policy). Default Keras reduction gives a scalar; a loss
            configured with no reduction can retain a leading batch dimension.

    Raises:
        ValueError: If no loss was compiled or inputs are nested structures.
        tf.errors.InvalidArgumentError: If loss inputs have incompatible shapes.
    """

    compiled_loss = model._compile_loss
    # A custom training step still requires an explicit compiled data loss.
    if compiled_loss is None:
        raise ValueError("Compile the model with a loss before evaluating it.")
    # These custom training steps pass one image/noise tensor pair per objective.
    if tf.nest.is_nested(y_true) or tf.nest.is_nested(y_pred):
        raise ValueError("compute_compiled_loss expects one target and prediction tensor.")
    # Resolve native aliases, output structures, reductions, and loss weights.
    if not compiled_loss.built:
        compiled_loss.build(y_true, y_pred)

    dtype = model.dtype_policy.variable_dtype
    # Keras exposes loss dtype as read-only; keep its two internal fields aligned.
    if compiled_loss.dtype != dtype:
        compiled_loss._dtype_policy = tf.keras.dtype_policies.get(dtype)
        compiled_loss._dtype = dtype
    for index, entry in enumerate(compiled_loss._flat_losses):
        # Leave matching objects untouched and never mutate a caller-owned loss.
        if entry.loss.dtype != dtype:
            loss_fn = copy(entry.loss)
            loss_fn._dtype_policy = tf.keras.dtype_policies.get(dtype)
            loss_fn._dtype = dtype
            compiled_loss._flat_losses[index] = entry._replace(loss=loss_fn)

    loss = compiled_loss(y_true, y_pred, sample_weight)
    for penalty in regularization_losses:
        loss = loss + model._aggregate_additional_loss(tf.cast(penalty, dtype))
    return loss


def display_name(name: object) -> str:
    """Render a name with the project's readable hierarchy separator.

    Args:
        name (object): Name or path converted to text before replacing slashes.

    Returns:
        formatted_name (str): Text with each slash replaced by ``__``.
            Other characters are retained; framework objects are not changed.

    Raises:
        None.
    """

    return str(name).replace("/", NAME_SEPARATOR)


def variable_path(variable: object) -> str:
    """Read a full variable path from Keras or a raw TensorFlow variable.

    Args:
        variable (object): Variable exposing Keras ``path`` or TensorFlow ``name``.

    Returns:
        path (str): Nonempty Keras path when available, otherwise the raw name.
            Native separators are retained for internal matching.

    Raises:
        AttributeError: If neither a nonempty path nor a name is available.
    """

    return getattr(variable, "path", None) or variable.name


def format_variable_name(variable: object) -> str:
    """Format both Keras variables and raw TensorFlow variables consistently.

    Args:
        variable (object): Variable exposing a Keras path or TensorFlow name.

    Returns:
        formatted_name (str): Full variable path using ``__`` between scopes.

    Raises:
        AttributeError: If the object exposes neither a usable path nor a name.
    """

    return display_name(variable_path(variable))


def optimizer_iterations(optimizer: object) -> object:
    """Read actual optimizer update counts through loss-scale wrappers.

    Args:
        optimizer (object): Ordinary optimizer, nested ``inner_optimizer``
            wrapper, or None for an absent optimizer.

    Returns:
        iterations (object): Innermost optimizer's integer iteration variable,
            typically an int64 Keras variable, or None when unavailable. Dynamic
            loss scaling can skip an update without advancing this counter.

    Raises:
        None.
    """

    while hasattr(optimizer, "inner_optimizer"):
        optimizer = optimizer.inner_optimizer

    return getattr(optimizer, "iterations", None)


def register_optimizer_variables(
    optimizer: tf.keras.optimizers.Optimizer, 
    variables: Sequence[object], 
    preserve_slot_prefixes: bool = False
) -> tf.keras.optimizers.Optimizer:
    """Return an optimizer covering ``variables`` without discarding old slots.

    Keras 3 optimizers cannot extend their variable registry after ``build``.
    Recreate only when a new variable appears, retaining matching optimizer
    state (including iterations, loss scaling, and existing moment estimates).
    Slots for a changed shape start at the optimizer's default unless prefix
    preservation is requested. This helper does not permit adding layers to an
    already built Keras model.

    Args:
        optimizer (tf.keras.optimizers.Optimizer): Keras optimizer whose
            configuration and compatible state must survive a variable change.
        variables (Sequence[object]): Complete variable selection to register.
            Duplicate objects are removed while retaining their first position.
        preserve_slot_prefixes (bool): Copy matching state into leading slices
            when every destination dimension is at least as large. Match slots
            by exact variable paths or unique paths differing only by a leading
            model scope. Newly added entries retain optimizer initialization.
            An unbuilt source remains unmodified. Defaults to False.

    Returns:
        registered_optimizer (tf.keras.optimizers.Optimizer): Original optimizer
            for empty/already registered selections, or a reconstructed optimizer
            with matching state copied using destination dtypes.
            Callers must retain the returned object when reconstruction occurs.

    Raises:
        ValueError: If optimizer state names/shapes are ambiguous, configuration
            cannot be reconstructed, or Keras rejects the selected variables.
    """

    variables = list(dict((id(v), v) for v in variables).values())

    # Empty phases need no optimizer registry.
    if not variables:
        return optimizer

    # Existing callers retain in-place initial optimizer registration.
    if not optimizer.built and not preserve_slot_prefixes:
        optimizer.build(variables)
        return optimizer

    # Prefix migration builds an independent candidate even from an unbuilt source.
    if optimizer.built:
        owner = getattr(optimizer, "inner_optimizer", optimizer)
        known = {id(v) for v in owner._trainable_variables}
        # Keep an existing instance when its selected variable objects are unchanged.
        if all(id(v) in known for v in variables):
            return optimizer

    replacement = type(optimizer).from_config(deepcopy(optimizer.get_config()))
    replacement.build(variables)


    def copy_state(source: object, destination: object) -> None:
        """Transfer unambiguous local optimizer state through nested wrappers.

        Args:
            source (object): Built optimizer with existing state variables.
            destination (object): Reconstructed optimizer with matching wrapper
                structure and the complete new variable selection.

        Returns:
            result (None): Matching state is assigned in destination variable
                dtypes; new slot entries retain their initialization.

        Raises:
            ValueError: If a source repeats a state name/shape combination or
                more than one source shape can supply a requested prefix.
        """

        source_inner = getattr(source, "inner_optimizer", None)
        old_values = source._variables if source_inner is not None else source.variables
        new_values = destination._variables if source_inner is not None else destination.variables
        old_state = {}
        old_by_name = {}
        for value in old_values:
            key = value.name, tuple(value.shape)

            # A repeated name and shape cannot identify one source slot unambiguously.
            if key in old_state:
                raise ValueError(f"Ambiguous optimizer state name: {key[0]}")

            old_state[key] = value
            old_by_name.setdefault(value.name, []).append(value)
        aliases = {}
        # Serialization may add/remove a Sequential scope before variable creation.
        if preserve_slot_prefixes:
            old_variables = getattr(source, "_trainable_variables", [])
            used_sources = set()
            for new_variable in getattr(destination, "_trainable_variables", []):
                new_path = variable_path(new_variable)
                matches = [
                    old_variable for old_variable in old_variables
                    if variable_path(old_variable) == new_path
                ]
                # Prefer exact identities before comparing model-relative paths.
                if not matches:
                    matches = [
                        old_variable for old_variable in old_variables
                        if variable_path(old_variable).endswith("/" + new_path)
                        or new_path.endswith("/" + variable_path(old_variable))
                    ]
                # A repeated relative path cannot identify a safe slot source.
                if len(matches) > 1:
                    raise ValueError(f"Ambiguous optimizer variable path: {new_path}")
                # Entirely new variables retain their initialized optimizer slots.
                if not matches:
                    continue
                old_variable = matches[0]
                # One learned variable must not initialize two unrelated replacements.
                if id(old_variable) in used_sources:
                    raise ValueError(f"Repeated optimizer variable source: {new_path}")
                used_sources.add(id(old_variable))
                old_prefix = variable_path(old_variable).replace("/", "_").replace(":", "_") + "_"
                new_prefix = new_path.replace("/", "_").replace(":", "_") + "_"
                for old_value in old_values:
                    # Keras slots append their role to the reference variable path.
                    if old_value.name.startswith(old_prefix):
                        name = new_prefix + old_value.name[len(old_prefix):]
                        bucket = aliases.setdefault(name, [])
                        # Keep a shared source state only once in each candidate set.
                        if all(candidate is not old_value for candidate in bucket):
                            bucket.append(old_value)

        for value in new_values:
            # Existing callers retain exact-name/shape matching and fresh resized slots.
            if not preserve_slot_prefixes:
                old_value = old_state.get((value.name, tuple(value.shape)))
                # Copy only fully compatible state in the default mode.
                if old_value is not None:
                    value.assign(old_value)
                continue

            # Grow unambiguous corresponding slots, retaining initialized tails.
            candidates = [
                candidate for candidate in aliases.get(value.name, old_by_name.get(value.name, []))
                if len(candidate.shape) == len(value.shape) and all(
                    old <= new for old, new in zip(candidate.shape, value.shape)
                )
            ]
            # A same-name collision cannot choose a reliable source prefix.
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous optimizer state prefix: {value.name}")
            # A unique growing shape receives only its learned leading slice.
            if candidates:
                old_value = candidates[0]
                # Scalars and unchanged slots copy without allocating prefix buffers.
                if tuple(old_value.shape) == tuple(value.shape):
                    value.assign(old_value)
                    continue
                prefix = tuple(slice(0, size) for size in old_value.shape)
                expanded = value.numpy()
                expanded[prefix] = old_value.numpy()
                value.assign(expanded)

        # Preserve inner slots independently of the wrapper's own counters.
        if source_inner is not None:
            copy_state(source_inner, destination.inner_optimizer)


    copy_state(optimizer, replacement)

    return replacement
