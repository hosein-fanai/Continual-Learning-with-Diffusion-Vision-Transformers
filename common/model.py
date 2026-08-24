"""Shared model factories, callbacks, optimizers, and weight-copy utilities."""

from __future__ import annotations

from tensorflow.keras import losses

from copy import deepcopy
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from common.config import Config
from common.dataloader import get_dataset_spec


_CLASSIFIER_MODELS = {"cnn", "dnn", "pretrained", "hp-tuned"}
_DIFFUSION_MODELS = {
    "diffusion_transformer", "dit_classifier", "dit_decoder", 
    "dit_encoder_decoder", "dit_encoder_decoder_classifier", 
    "unet", "unet_classifier"
}
_VAE_MODELS = {"vae", "variational_autoencoder", "vae_classifier"}


def get_compile_args(
    optimizer: object = "adam", 
    metrics: Sequence[object] = ("accuracy",), 
    loss: str | losses.Loss | Callable = "sparse_categorical_crossentropy"
) -> dict[str, object]:
    """Build the standard keyword mapping used by ``keras.Model.compile``.

    Args:
        optimizer (str | tf.keras.optimizers.Optimizer): Optimizer identifier or
            instance accepted by Keras.  The default is ``"adam"``.
        metrics (list[str | tf.keras.metrics.Metric]): Metrics evaluated by
            Keras.  ``["accuracy"]`` selects accuracy appropriate to the loss
            and target representation; ``[]`` disables extra metrics.
        loss (str | tf.keras.losses.Loss | Callable): Keras loss identifier,
            object, or callable.  The default expects integer class labels.

    Returns:
        dict[str, object]: A new mapping with exactly ``optimizer``, ``loss``,
        and ``metrics`` keys, suitable for ``model.compile(**result)``.
    """

    compile_args = {
        "optimizer": optimizer, 
        "loss": loss, 
        "metrics": metrics
    }

    return compile_args


def get_callbacks(
    indices: Sequence[int] = (0,), 
    monitor: str = "val_accuracy", 
    mode: str = "max", 
    patience: int = 5, 
    min_delta: float = 1e-2, 
    reducelr_factor: float = 0.6, 
    verbose: int = 1
) -> list[object]:
    """Construct selected early-stopping and learning-rate callbacks.

    Args:
        indices (Sequence[int]): Positions selected from ``[EarlyStopping,
            ReduceLROnPlateau]``.  ``[0]`` returns only early stopping, ``[1]``
            only learning-rate reduction, and ``[0, 1]`` both.  Normal Python
            negative indexing and duplicates are honored.
        monitor (str): Key expected in epoch logs, such as ``"val_accuracy"``,
            ``"val_loss"``, or the project's ``"decoder_accuracy"``.
        mode (str): ``"min"``, ``"max"``, or ``"auto"`` direction used by
            ``EarlyStopping``.  This argument is not forwarded to
            ``ReduceLROnPlateau``, whose Keras default remains ``"auto"``.
        patience (int): Number of non-improving epochs tolerated by both
            callbacks.
        min_delta (float): Minimum absolute change counted as improvement by
            both callbacks.
        reducelr_factor (float): Multiplier applied by ``ReduceLROnPlateau``;
            normally a float strictly between 0 and 1.
        verbose (int): Keras callback verbosity, commonly ``0`` or ``1``.

    Returns:
        list[tf.keras.callbacks.Callback]: Newly created callbacks in the exact
        order requested by ``indices``.  Early stopping restores best weights.

    Raises:
        IndexError: If an index does not address either of the two callbacks.
    """

    from tensorflow.keras import callbacks


    callbacks_list = [
        callbacks.EarlyStopping(
            monitor=monitor, 
            mode=mode, 
            restore_best_weights=True, 
            patience=patience, 
            min_delta=min_delta, 
            verbose=verbose
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor, 
            patience=patience, 
            min_delta=min_delta, 
            factor=reducelr_factor, 
            verbose=verbose
        )
    ]

    return [callbacks_list[idx] for idx in indices]


def _make_optimizer(config: Config | None = None, 
                    **kwargs: object) -> object:
    """Build a TensorFlow 2.10 optimizer and optional learning-rate schedule.

    Args:
        config (Config | None): Typed optimizer/training settings.  When
            supplied, direct keyword values are ignored.
        **kwargs (object): Direct-mode values ``epochs``,
            ``initial_learning_rate``, ``decay_steps``, ``name``, ``schedule``,
            ``weight_decay``, ``momentum``, ``clipnorm``, and ``trainset_len``.
            Direct mode defaults to a constant learning rate because no
            dataset length is necessarily available.

    Returns:
        tf.keras.optimizers.Optimizer | object: The requested Keras optimizer,
        or a non-string optimizer object passed by the caller unchanged.

    Raises:
        ValueError: If a schedule/optimizer is unsupported, cosine decay lacks
            a positive duration, or weight decay is requested for a TensorFlow
            2.10 optimizer other than experimental AdamW.
    """

    from tensorflow.keras import optimizers


    # Resolve optimizer settings from direct keyword arguments.
    if config is None:
        epochs = kwargs.get("epochs", 10)
        initial_learning_rate = kwargs.get("initial_learning_rate", 5e-3)
        decay_steps = kwargs.get("decay_steps")
        name = kwargs.get("name", "adam")
        schedule = kwargs.get("schedule", "constant")
        weight_decay = kwargs.get("weight_decay")
        momentum = kwargs.get("momentum", 0.)
        clipnorm = kwargs.get("clipnorm")
        trainset_len = kwargs.get("trainset_len", None)
    # Resolve optimizer settings from typed project configuration.
    else:
        epochs = config.training.epochs
        initial_learning_rate = config.optimizer.initial_learning_rate
        decay_steps = config.optimizer.decay_steps
        name = config.optimizer.name
        schedule = config.optimizer.schedule
        weight_decay = config.optimizer.weight_decay
        momentum = config.optimizer.momentum
        clipnorm = config.optimizer.clipnorm
        trainset_len = config.dataset.trainset_len

    # Preserve an optimizer object supplied directly by the caller.
    if not isinstance(name, str):
        return name

    schedule = schedule.lower() if isinstance(schedule, str) else schedule
    # Derive a cosine duration from epochs and prepared dataset length.
    if schedule == "cosine" and decay_steps is None:
        # Require dataset sizing when cosine duration cannot be inferred otherwise.
        if trainset_len is None:
            raise ValueError(
                "trainset_len is required when decay_steps is not provided."
            )

        decay_steps = epochs * trainset_len
        # Record the resolved cosine duration in typed configuration.
        if config is not None:
            config.optimizer.decay_steps = decay_steps

    # Require a positive integral duration for cosine decay.
    if schedule == "cosine" and (not isinstance(decay_steps, int) \
    or decay_steps <= 0):
        raise ValueError("decay_steps must be a positive integer for cosine decay.")

    learning_rate = initial_learning_rate
    # Construct the requested cosine learning-rate schedule.
    if schedule == "cosine":
        learning_rate = optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate, 
            decay_steps=decay_steps
        )
    # Reject schedule modes outside constant and cosine decay.
    elif schedule not in ("constant", None):
        raise ValueError(
            "schedule must be None, 'cosine', or 'constant'."
        )

    name = name.lower()
    optimizer_kwargs = {"learning_rate": learning_rate}
    # Forward optional gradient clipping to the optimizer.
    if clipnorm is not None:
        optimizer_kwargs["clipnorm"] = clipnorm

    # Construct the TensorFlow 2.10 Adam optimizer.
    if name == "adam":
        # Prevent unsupported Adam weight decay from being silently ignored.
        if weight_decay not in (None, 0, 0.):
            raise ValueError("weight_decay requires optimizer name 'adamw'.")
        return optimizers.Adam(**optimizer_kwargs)

    # Construct experimental AdamW when explicitly requested.
    if name == "adamw":
        adamw = getattr(optimizers, "AdamW", None)
        # Report TensorFlow installations that do not expose AdamW.
        if adamw is None:
            adamw = optimizers.experimental.AdamW

        return adamw(
            weight_decay=0. if weight_decay is None else weight_decay,
            **optimizer_kwargs
        )

    # Construct the TensorFlow 2.10 Nadam optimizer.
    if name == "nadam":
        # Prevent unsupported Nadam weight decay from being silently ignored.
        if weight_decay not in (None, 0, 0.):
            raise ValueError("weight_decay requires optimizer name 'adamw'.")
        return optimizers.Nadam(**optimizer_kwargs)

    # Construct RMSprop with its configured momentum.
    if name == "rmsprop":
        # Prevent unsupported RMSprop weight decay from being silently ignored.
        if weight_decay not in (None, 0, 0.):
            raise ValueError("weight_decay requires optimizer name 'adamw'.")
        return optimizers.RMSprop(
            momentum=momentum, 
            **optimizer_kwargs
        )

    # Construct SGD with its configured momentum.
    if name == "sgd":
        # Prevent unsupported SGD weight decay from being silently ignored.
        if weight_decay not in (None, 0, 0.):
            raise ValueError("weight_decay requires optimizer name 'adamw'.")
        return optimizers.SGD(
            momentum=momentum, 
            **optimizer_kwargs
        )

    raise ValueError(
        "Unsupported optimizer name: " + str(name)
    )


def _get_classifier_model(
    class_num: int, 
    model_type: str = "CNN", 
    model_path: str = "", 
    dropout_rate: float = 0., 
    num_last_not_frozen: int = 3, 
    resize: tuple[int, int] = (299, 299), 
    compile_args: Mapping[str, object] | None = None, 
    use_loaded_opt: bool = False, 
    verbose: bool | int = 1, 
    architecture_kwargs: Mapping[str, object] | None = None
) -> Any:
    """Build and compile one of four legacy image/feature classifiers.

    Args:
        class_num (int): Positive output class count and final softmax width.
        model_type (str): Exactly one of ``"pretrained"``, ``"hp-tuned"``,
            ``"CNN"``, or ``"DNN"``.  ``pretrained`` resizes ``32x32x3`` images
            and uses Xception; ``CNN`` consumes ``32x32x3`` images; ``DNN``
            consumes 2,048-element feature vectors; ``hp-tuned`` clones a saved
            model's architecture except for its original output layer.
        model_path (str | os.PathLike): Keras model path required by
            ``"hp-tuned"`` and ignored by other model types.
        dropout_rate (float): Fraction dropped before the final classifier;
            normally in ``[0, 1)``.
        num_last_not_frozen (int): For Xception, the number of trailing layers
            left trainable.  Earlier layers are frozen.  A value of ``0``
            freezes none because Python's ``layers[:-0]`` slice is empty.
        resize (tuple[int, int]): ``(height, width)`` used to resize images and
            define Xception's input size.  Xception imposes its own minimum-size
            requirements.
        compile_args (Mapping[str, object]): Overrides or extends defaults
            ``{"optimizer": "adam", "loss":
            "sparse_categorical_crossentropy", "metrics": ["accuracy"]}``.
            Valid additional keys are those accepted by ``Model.compile``, for
            example ``{"optimizer": Adam(1e-4), "run_eagerly": True}``.
        use_loaded_opt (bool): In ``"hp-tuned"`` mode, replace any requested
            optimizer with the optimizer deserialized from ``model_path``.
            Ignored by other modes.
        verbose (bool | int): Truthy values print ``model.summary()``.
        architecture_kwargs (Mapping[str, object]): Optional architecture
            controls for ``CNN`` or ``DNN``.  An empty mapping preserves the
            original models exactly.  CNN controls are ``input_shape``
            (default ``(32, 32, 3)``), ``conv_filters`` (``(64, 128, 128,
            256)``), ``conv_depths`` (``(2, 2, 2, 1)``), ``kernel_size`` (3),
            ``first_kernel_size`` (7), ``activation`` (``"relu"``),
            ``use_batch_norm`` (false), ``pooling`` (``"max"``), and
            ``global_pooling`` (``"avg"``).  DNN controls are ``input_shape``
            (default ``(2048,)``), ``hidden_dims`` (empty), ``activation``
            (``"relu"``), ``use_batch_norm`` (false), and
            ``kernel_initializer`` (``"glorot_uniform"``).  Multidimensional
            DNN inputs are flattened before the dense blocks.  Nonempty
            mappings are rejected for ``pretrained`` and ``hp-tuned`` models.

    Returns:
        tf.keras.Sequential: A built, compiled classifier mapping a batch of
        images/features to ``float32`` probabilities shaped
        ``[batch, class_num]``.

    Raises:
        TypeError: If ``architecture_kwargs`` contains an unsupported key.
        ValueError: If a size/rate is invalid, an hp-tuned model path is empty,
            CNN filter/depth lengths differ, or a pooling/model option is
            unsupported.

    Note:
        ``hp-tuned`` clones layer configuration, not the loaded layer weights;
        :func:`copy_model` is used separately when prior weights are required.
    """

    from tensorflow.keras import models, layers, applications


    # Require a positive non-boolean classifier output width.
    if not isinstance(class_num, Integral) or isinstance(class_num, bool) \
    or class_num <= 0:
        raise ValueError("class_num must be a positive integer.")
    # Require a numeric dropout rate rather than a boolean flag.
    if isinstance(dropout_rate, bool) or not isinstance(dropout_rate, Real):
        raise TypeError("dropout_rate must be a real number.")
    # Keep dropout probability within its mathematical domain.
    if not 0. <= dropout_rate < 1.:
        raise ValueError("dropout_rate must lie in [0, 1).")
    # Require a nonnegative integral fine-tuning depth.
    if not isinstance(num_last_not_frozen, Integral) \
    or isinstance(num_last_not_frozen, bool) or num_last_not_frozen < 0:
        raise ValueError("num_last_not_frozen must be a nonnegative integer.")
    # Require two positive resize dimensions.
    if not isinstance(resize, Sequence) or len(resize) != 2 or any(
        not isinstance(size, Integral) or isinstance(size, bool) or size <= 0
        for size in resize
    ):
        raise ValueError("resize must contain two positive integers.")

    class_num = int(class_num)
    dropout_rate = float(dropout_rate)
    num_last_not_frozen = int(num_last_not_frozen)
    resize = tuple(int(size) for size in resize)

    compile_args_default = get_compile_args()
    compile_args = {
        **compile_args_default, 
        **(compile_args or {})
    }
    architecture_kwargs = dict(architecture_kwargs or {})

    # Keep custom local architectures separate from saved/pretrained models.
    if architecture_kwargs and model_type in ("pretrained", "hp-tuned"):
        raise ValueError(
            "architecture_kwargs is only supported for CNN and DNN models."
        )
    # Require a saved model path for the tuned-model family.
    if model_type == "hp-tuned" and not model_path:
        raise ValueError("model_path is required for an hp-tuned classifier.")

    # Build an ImageNet Xception transfer-learning classifier.
    if model_type == "pretrained":
        conv_base = applications.Xception(
            include_top=False, 
            input_shape=(resize[0], resize[1], 3)
        )
        for layer in conv_base.layers[:-num_last_not_frozen]:
            layer.trainable = False

        model = models.Sequential([
            layers.Resizing(
                resize[0],
                resize[1],
                input_shape=(32, 32, 3), 
                name="resize"
            ), 
            layers.Rescaling(
                scale=1. / 127.5,
                offset=-1.,
                name="xception_preprocess"
            ), 
            conv_base, 
            layers.GlobalAveragePooling2D(), 
            layers.Dropout(dropout_rate), 
            layers.Dense(class_num, activation="softmax")
        ])
    # Restore the requested hyperparameter-tuned classifier.
    elif model_type == "hp-tuned":
        model = models.load_model(model_path)
        # Reuse the deserialized optimizer when requested.
        if use_loaded_opt:
            compile_args["optimizer"] = model.optimizer

        model = models.Sequential([
            *models.clone_model(model).layers[:-1], 
            layers.Dense(class_num, activation="softmax")
        ])
    # Build a configurable convolutional classifier.
    elif model_type == "CNN":
        # Preserve the compact legacy CNN when no architecture is supplied.
        if not architecture_kwargs:
            model = models.Sequential([
                layers.Conv2D(64, 7, padding="same", activation="relu", input_shape=(32, 32, 3)), 
                layers.Conv2D(64, 3, padding="same", activation="relu"), 
                layers.MaxPooling2D(2), 
                layers.Conv2D(128, 3, padding="same", activation="relu"), 
                layers.Conv2D(128, 3, padding="same", activation="relu"), 
                layers.MaxPooling2D(2), 
                layers.Conv2D(128, 3, padding="same", activation="relu"), 
                layers.Conv2D(128, 3, padding="same", activation="relu"), 
                layers.MaxPooling2D(2), 
                layers.Conv2D(256, 3, padding="same", activation="relu"), 
                layers.GlobalAveragePooling2D(), 
                layers.Dropout(dropout_rate), 
                layers.Dense(class_num, activation="softmax")
            ])
        # Merge user overrides into explicit CNN architecture defaults.
        else:
            cnn_defaults = {
                "input_shape": (32, 32, 3), 
                "conv_filters": (64, 128, 128, 256), 
                "conv_depths": (2, 2, 2, 1), 
                "kernel_size": 3, 
                "first_kernel_size": 7, 
                "activation": "relu", 
                "use_batch_norm": False, 
                "pooling": "max", 
                "global_pooling": "avg"
            }
            unknown = sorted(set(architecture_kwargs) - set(cnn_defaults))
        # Reject unrecognized CNN architecture keys.
            if unknown:
                raise TypeError("Unsupported CNN architecture options: " + str(unknown))
            architecture = {
                **cnn_defaults, 
                **architecture_kwargs
            }
            conv_filters = architecture["conv_filters"]
            conv_depths = architecture["conv_depths"]

        # Keep convolutional widths aligned with their stage depths.
            if len(conv_filters) != len(conv_depths):
                raise ValueError("conv_filters and conv_depths must have equal lengths.")
        # Restrict intermediate pooling to max or average pooling.
            if architecture["pooling"] not in ("max", "avg"):
                raise ValueError("pooling must be 'max' or 'avg'.")
        # Restrict final spatial reduction to max or average pooling.
            if architecture["global_pooling"] not in ("max", "avg"):
                raise ValueError("global_pooling must be 'max' or 'avg'.")

            pooling_layer = layers.MaxPooling2D if architecture["pooling"] == "max" \
                            else layers.AveragePooling2D
            global_pooling_layer = layers.GlobalAveragePooling2D \
                                if architecture["global_pooling"] == "avg" \
                                else layers.GlobalMaxPooling2D
            model_layers = [
                layers.InputLayer(input_shape=architecture["input_shape"])
            ]

            for stage_id, (filters, depth) in enumerate(zip(
                conv_filters, conv_depths
            )):
                for block_id in range(depth):
                    model_layers.append(layers.Conv2D(
                        filters, 
                        architecture["first_kernel_size"] if stage_id == block_id == 0
                        else architecture["kernel_size"], 
                        padding="same", 
                        activation=architecture["activation"]
                    ))
            # Normalize convolutional activations when configured.
                    if architecture["use_batch_norm"]:
                        model_layers.append(layers.BatchNormalization())

            # Downsample between convolutional stages, not after the final stage.
                if stage_id < len(conv_filters) - 1:
                    model_layers.append(pooling_layer(2))

            model_layers.extend([
                global_pooling_layer(), 
                layers.Dropout(dropout_rate), 
                layers.Dense(class_num, activation="softmax")
            ])
            model = models.Sequential(model_layers)
    # Build a configurable dense classifier.
    elif model_type == "DNN":
        # Preserve the compact legacy DNN when no architecture is supplied.
        if not architecture_kwargs:
            model = models.Sequential([
                # layers.Flatten(input_shape=(10, 10, 2048)), 
                # layers.GlobalAveragePooling2D(input_shape=(10, 10, 2048)), 
                # layers.Dense(256, activation="relu"), 
                layers.Dropout(dropout_rate, input_shape=(2048,)), 
                layers.Dense(class_num, activation="softmax")
            ])
        # Merge user overrides into explicit DNN architecture defaults.
        else:
            dnn_defaults = {
                "input_shape": (2048,), 
                "hidden_dims": (), 
                "activation": "relu", 
                "use_batch_norm": False, 
                "kernel_initializer": "glorot_uniform"
            }
            unknown = sorted(set(architecture_kwargs) - set(dnn_defaults))
        # Reject unrecognized DNN architecture keys.
            if unknown:
                raise TypeError("Unsupported DNN architecture options: " + str(unknown))
            architecture = {
                **dnn_defaults, 
                **architecture_kwargs
            }

            input_shape = tuple(architecture["input_shape"])
            model_layers = [layers.InputLayer(input_shape=input_shape)]
        # Flatten structured inputs before dense hidden layers.
            if len(input_shape) > 1:
                model_layers.append(layers.Flatten())

            for hidden_dim in architecture["hidden_dims"]:
                model_layers.append(layers.Dense(
                    hidden_dim, 
                    activation=architecture["activation"], 
                    kernel_initializer=architecture["kernel_initializer"]
                ))
            # Normalize hidden activations when configured.
                if architecture["use_batch_norm"]:
                    model_layers.append(layers.BatchNormalization())

            model_layers.extend([
                layers.Dropout(dropout_rate), 
                layers.Dense(
                    class_num, 
                    activation="softmax", 
                    kernel_initializer=architecture["kernel_initializer"]
                )
            ])
            model = models.Sequential(model_layers)
    # Reject classifier families outside the supported implementations.
    else:
        raise ValueError(
            "model_type needs to be one of pretrained, hp-tuned, CNN, or DNN."
        )

    model.compile(**compile_args)
    model.build(model.layers[0].input_shape)

    # Print the constructed classifier architecture when requested.
    if verbose:
        model.summary()

    return model


def get_model(
    config: Config | dict[str, object] | int | None = None, 
    **kwargs: object
) -> Any | dict[str, object]:
    """Build any classifier, VAE, or diffusion model used by the project.

    Pass a :class:`common.config.Config` object for configured experiments, or
    pass the same settings directly as keyword arguments.  The original
    ``get_model(class_num, model_type=...)`` classifier API remains supported.

    Args:
        config (Config | int | dict[str, object] | None): A complete config,
            legacy positional class count, compatible root mapping, or ``None``
            for direct keywords.
        **kwargs (object): Direct selections such as ``model_name``/``name``,
            ``model_kwargs``, ``wrapper_name``, ``wrapper_kwargs``,
            ``classifier_name``, ``classifier_kwargs``, dataset shape/count
            values (including raw-image ``pad``), optimizer values, ``task``,
            ``loss_function``, summary/weight settings, and the documented
            legacy classifier options. Typed configured VAE/diffusion sections
            inherit dataset dimensions and class count.

    Returns:
        tf.keras.Model | dict[str, object]: A built and compiled classifier,
            VAE, or diffusion wrapper. Continual tasks return ``classifier``,
            ``classifier_name``, and ``generative_model``. A classifier-family
            ``model.name`` creates a classifier-only bundle; buffer replay and
            classifier-only bundles set ``generative_model`` to ``None``.
            Configured weights initialize the replay model when present,
            otherwise the returned classifier.

    Raises:
        TypeError: If positional/direct options conflict, a model option is
            unsupported, or ``pad`` is not a non-boolean integer.
        ValueError: If a model, wrapper, optimizer, schedule, or architecture
            selection is unsupported, lacks required dataset sizing, or
            ``pad`` is negative/incompatible with the selected input type.
    """

    # Preserve the legacy positional class-count API.
    if isinstance(config, int):
        # Reject duplicate positional and keyword class counts.
        if "class_num" in kwargs:
            raise TypeError("class_num was supplied both positionally and by keyword.")
        kwargs = {"class_num": config, **kwargs}
        config = None

    # Convert compatible configuration mappings to typed configuration.
    if isinstance(config, dict):
        config = Config(**config)

    # Reject unsupported configuration root types.
    if config is not None and not isinstance(config, Config):
        raise TypeError("config must be a Config, mapping, integer class count, or None.")
  
    legacy_keys = {
        "class_num", "model_type", "model_path", "dropout_rate", 
        "num_last_not_frozen", "resize", "compile_args", 
        "use_loaded_opt", "verbose", "architecture_kwargs"
    }
    # Route calls using only legacy classifier options to the original builder.
    if config is None and "class_num" in kwargs and set(kwargs) <= legacy_keys:
        legacy_type = kwargs.get("model_type", "CNN")
        # Use legacy construction only for classifier families.
        if str(legacy_type).lower() in _CLASSIFIER_MODELS:
            legacy_kwargs = dict(kwargs)
            class_num = legacy_kwargs.pop("class_num")
            legacy_kwargs["model_type"] = _classifier_name(
                legacy_kwargs.get("model_type", "CNN")
            )

            return _get_classifier_model(class_num, **legacy_kwargs)


    from autoencoder import VAEClassifier, VariationalAutoencoder

    from diffusion import (
        UNet, 
        UNetClassifier, 
        DiTClassifier, 
        DiTDecoder, 
        DiTEncoderDecoder, 
        DiTEncoderDecoderClassifier, 
        DiffusionTransformer, 
        DiffusionClassifier, 
        DiffusionClassifierV2, 
        DiffusionModel
    )


    using_typed_model_config = False
    # Resolve model settings from direct keyword arguments.
    if config is None:
        default_model_name = "dit_classifier" if kwargs.get(
            "with_classifier", True
        ) else "diffusion_transformer"
        model_name = kwargs.get(
            "model_name", kwargs.get("model_type", kwargs.get("name"))
        ) or default_model_name
        dataset_name = kwargs.get("dataset_name", "mnist")
        return_features = kwargs.get("return_features", False)
        pad = kwargs.get("pad", 0)
        class_num, image_shape, flat_dim = get_dataset_spec(
            dataset_name, 
            return_features
        )
        class_num = kwargs.get("class_num", class_num)
        image_shape = tuple(kwargs.get("image_shape", image_shape))
        flat_dim = kwargs.get("flat_dim", flat_dim)
        model_kwargs = deepcopy(
            kwargs.get("model_kwargs", kwargs.get("kwargs", {}))
        )
        wrapper_name = kwargs.get("wrapper_name")
        wrapper_kwargs = deepcopy(kwargs.get("wrapper_kwargs", {}))
        classifier_name = kwargs.get("classifier_name")
        classifier_kwargs = deepcopy(kwargs.get("classifier_kwargs", {}))
        trainset_len = kwargs.get("trainset_len")
        onehot_labels = kwargs.get("onehot_labels", False)
        task = kwargs.get("task", "legacy")
        loss_function = kwargs.get("loss_function", "mse")
        show_network_summary = kwargs.get("show_network_summary", False)
        weights_path = kwargs.get("weights_path")

        for key in (
            "model_path", "dropout_rate", "num_last_not_frozen", "resize", 
            "compile_args", "use_loaded_opt", "architecture_kwargs"
        ):
            # Forward documented classifier shortcuts into model options.
            if key in kwargs:
                model_kwargs[key] = kwargs[key]
    # Resolve model settings from typed project configuration.
    else:
        dataset_name = config.dataset.name
        return_features = config.dataset.return_features
        pad = config.dataset.pad
        class_num, image_shape, flat_dim = get_dataset_spec(
            dataset_name, 
            return_features
        )
        trainset_len = config.dataset.trainset_len
        onehot_labels = config.dataset.onehot_labels
        task = config.training.task
        loss_function = config.model.loss_function
        # Restrict continual output width to the requested leading classes.
        if task.lower() == "continual" \
        and config.continually_learn.class_num is not None:
            continual_class_num = config.continually_learn.class_num
            # Keep continual class count within dataset bounds.
            if not 2 <= continual_class_num <= class_num:
                raise ValueError(
                    "continually_learn.class_num must be between "
                    "2 and the selected dataset's class count."
                )
            class_num = continual_class_num
        show_network_summary = config.model.show_network_summary
        weights_path = config.model.weights_path
        classifier_name = config.model.classifier_name
        classifier_kwargs = deepcopy(config.model.classifier_kwargs)

        # Preserve compact legacy typed diffusion selection when name is omitted.
        if config.model.name is None:
            using_typed_model_config = True
            # Select the legacy classifier-capable diffusion path.
            if config.model.with_classifier:
                model_name = "dit_classifier"
                model_kwargs = config.model.dit_classifier.kwargs()
                wrapper_name = "diffusion_classifier"
                wrapper_kwargs = config.model.diffusion_classifier.kwargs()
            # Select the legacy generator-only diffusion path.
            else:
                model_name = "diffusion_transformer"
                model_kwargs = config.model.diffusion_transformer.kwargs()
                wrapper_name = "diffusion_model"
                wrapper_kwargs = config.model.diffusion_model.kwargs()
        # Resolve an explicitly named configured model family.
        else:
            model_name = config.model.name
            wrapper_name = config.model.wrapper_name
            # Give generic model options precedence when explicitly supplied.
            if config.model.kwargs:
                model_kwargs = deepcopy(config.model.kwargs)
            # Otherwise obtain options from the matching typed model section.
            else:
                section_names = {
                    "diffusion_transformer": "diffusion_transformer",
                    "dit_classifier": "dit_classifier",
                    "dit_decoder": "dit_decoder",
                    "dit_encoder_decoder": "dit_encoder_decoder",
                    "dit_encoder_decoder_classifier": (
                        "dit_encoder_decoder_classifier"
                    ),
                    "unet": "unet",
                    "unet_classifier": "unet_classifier",
                    "vae": "variational_autoencoder",
                    "variational_autoencoder": "variational_autoencoder",
                    "vae_classifier": "vae_classifier",
                }
                section_name = section_names.get(str(model_name).lower())
                using_typed_model_config = section_name is not None
                model_kwargs = getattr(config.model, section_name).kwargs() \
                    if section_name is not None else {}

            # Give generic wrapper options precedence when explicitly supplied.
            if config.model.wrapper_kwargs:
                wrapper_kwargs = deepcopy(config.model.wrapper_kwargs)
            # Otherwise obtain options from the matching typed wrapper section.
            else:
                normalized_name = str(model_name).lower()
                default_wrapper_name = "diffusion_classifier" if normalized_name in {
                    "dit_classifier",
                    "dit_encoder_decoder_classifier",
                    "unet_classifier",
                } else "diffusion_model"
                wrapper_section_name = str(
                    wrapper_name or default_wrapper_name
                ).lower() if normalized_name in _DIFFUSION_MODELS else None
                wrapper_section = getattr(
                    config.model, wrapper_section_name, None
                ) if wrapper_section_name is not None else None
                wrapper_kwargs = wrapper_section.kwargs() \
                    if wrapper_section is not None else {}

    # Reject booleans and non-integral padding widths before shape changes.
    if isinstance(pad, (bool, np.bool_)) or not isinstance(pad, Integral):
        raise TypeError("pad must be a non-boolean integer.")
    # Keep spatial padding nonnegative.
    if pad < 0:
        raise ValueError("pad must be nonnegative.")

    task = task.lower()

    # Prevent image padding from being applied to saved feature vectors.
    if pad and return_features:
        raise ValueError("pad is not supported for saved feature inputs.")
    # Keep pretrained image geometry unchanged.
    if pad and model_name.lower() in {"pretrained", "hp-tuned"}:
        raise ValueError("pad is not supported for pretrained/hp-tuned models.")
    # Propagate raw-image padding into model input dimensions.
    if pad > 0:
        image_shape = (
            image_shape[0] + 2 * pad, 
            image_shape[1] + 2 * pad, 
            image_shape[2]
        )
        flat_dim = image_shape[0] * image_shape[1] * image_shape[2]

    model_name = model_name.lower()
    optimizer_options = dict(kwargs)
    optimizer_options.pop("trainset_len", None)


    def build_classifier(
        name: str, 
        options: Mapping[str, object]
    ) -> Any:
        """Build one configured standalone classifier.

        Args:
            name (str): Normalized classifier family name.
            options (Mapping[str, object]): Classifier constructor overrides.

        Returns:
            tf.keras.Model: Built and compiled classifier.
        """

        options = deepcopy(options)
        options.pop("class_num", None)
        model_path = options.pop("model_path", "")
        dropout_rate = options.pop("dropout_rate", 0.)
        num_last_not_frozen = options.pop("num_last_not_frozen", 3)
        resize = tuple(options.pop("resize", (299, 299)))
        architecture_kwargs = deepcopy(
            options.pop("architecture_kwargs", {}) or {}
        )
        compile_args = deepcopy(options.pop("compile_args", {}) or {})
        use_loaded_opt = options.pop("use_loaded_opt", False)

        # Reject classifier options not consumed by this builder.
        if options:
            raise TypeError(
                f"Unsupported {name} model options: {sorted(options)}"
            )

        # Supply the resolved flattened input width to dense classifiers.
        if name == "dnn":
            architecture_kwargs = {
                "input_shape": (flat_dim,), 
                **architecture_kwargs
            }
        # Supply the resolved image shape to convolutional classifiers.
        elif name == "cnn":
            architecture_kwargs = {
                "input_shape": image_shape, 
                **architecture_kwargs
            }
        # Enforce the three-channel input contract of pretrained Xception.
        elif name == "pretrained" and image_shape[-1] != 3:
            raise ValueError(
                "The pretrained Xception classifier requires three-channel inputs."
            )

        loss = "categorical_crossentropy" if onehot_labels else \
            "sparse_categorical_crossentropy"
        compile_args = {
            "optimizer": _make_optimizer(
                config, 
                trainset_len=trainset_len, 
                **optimizer_options
            ), 
            "loss": loss, 
            "metrics": ["accuracy"], 
            **compile_args
        }

        return _get_classifier_model(
            class_num, 
            model_type=_classifier_name(name), 
            model_path=model_path, 
            dropout_rate=dropout_rate, 
            num_last_not_frozen=num_last_not_frozen, 
            resize=resize, 
            compile_args=compile_args, 
            use_loaded_opt=use_loaded_opt, 
            verbose=0, 
            architecture_kwargs=architecture_kwargs
        )


    def build_selected(name: str) -> Any:
        """Build the selected raw family and, when needed, its wrapper.

        Args:
            name (str): Normalized model-family name.

        Returns:
            tf.keras.Model: Built and compiled selected model.
        """

        selected_kwargs = deepcopy(model_kwargs)
        # Delegate standalone classifier families to their shared builder.
        if name in _CLASSIFIER_MODELS:
            return build_classifier(name, selected_kwargs)

        optimizer = _make_optimizer(
            config, 
            trainset_len=trainset_len, 
            **optimizer_options
        )
        # Construct the plain variational-autoencoder family.
        if name in ("vae", "variational_autoencoder"):
            conditioned = selected_kwargs.pop("conditioned", True)
            # Require class conditioning for generative replay.
            if task == "continual" and not conditioned:
                # Supply the required conditioning automatically in typed mode.
                if using_typed_model_config:
                    conditioned = True
                # Respect direct explicit input by rejecting incompatible mode.
                else:
                    raise ValueError("Continual VAE replay requires conditioned=True.")

            selected_kwargs.pop("class_num", None)
            selected_kwargs.pop("compile", None)
            vae_compile_args = deepcopy(
                selected_kwargs.pop("compile_args", {}) or {}
            )

            # Make dataset-derived width authoritative for typed sections.
            if using_typed_model_config:
                selected_kwargs["data_dim"] = flat_dim
            # Preserve a direct data width while providing a dataset default.
            else:
                selected_kwargs.setdefault("data_dim", flat_dim)

            model = VariationalAutoencoder(
                conditioned=conditioned, 
                class_num=class_num if conditioned else None, 
                compile=False, 
                **selected_kwargs
            )
            model.compile(**{
                "optimizer": optimizer, 
                "loss": loss_function, 
                **vae_compile_args
            })

            return model

        # Construct the joint VAE-classifier family.
        if name == "vae_classifier":
            selected_kwargs.pop("conditioned", None)
            selected_kwargs.pop("class_num", None)
            selected_kwargs.pop("compile", None)

            # Make dataset-derived width authoritative for typed sections.
            if using_typed_model_config:
                selected_kwargs["data_dim"] = flat_dim
            # Preserve a direct data width while providing a dataset default.
            else:
                selected_kwargs.setdefault("data_dim", flat_dim)

            vae_compile_args = deepcopy(
                selected_kwargs.pop("compile_args", {}) or {}
            )
            vae_compile_args = {
                "optimizer": optimizer, 
                "loss": loss_function, 
                **vae_compile_args
            }
            selected_classifier_name = classifier_name or "dnn"
            classifier = build_classifier(
                selected_classifier_name.lower(), 
                classifier_kwargs
            )

            return VAEClassifier(
                class_num=class_num, 
                classifier=classifier, 
                compile_args=vae_compile_args, 
                **selected_kwargs
            )

        # Reject model names outside classifier, VAE, and diffusion families.
        if name not in _DIFFUSION_MODELS:
            raise ValueError("Unsupported model type: " + name)

        diffusion_compile_args = deepcopy(
            selected_kwargs.pop("compile_args", {}) or {}
        )
        dataset_dimensions = {
            "num_classes": None if using_typed_model_config and
                        selected_kwargs.get("num_classes") is None 
                        else class_num, 
            "image_size": image_shape[0], 
            "channels": image_shape[-1]
        }

        # Make dataset-derived diffusion dimensions authoritative in typed mode.
        if using_typed_model_config:
            selected_kwargs.update(dataset_dimensions)
        # Preserve direct dimensions while supplying missing dataset defaults.
        else:
            for key, value in dataset_dimensions.items():
                selected_kwargs.setdefault(key, value)

        # Construct the base diffusion transformer.
        if name == "diffusion_transformer":
            network = DiffusionTransformer(**selected_kwargs)
        # Construct a classification-capable diffusion transformer.
        elif name == "dit_classifier":
            network = DiTClassifier(**selected_kwargs)
        # Construct a standalone decoder with inferred encoder geometry.
        elif name == "dit_decoder":
            # Reject encoder aggregation options without an attached encoder.
            if selected_kwargs.get("feature_aggregation_ids_dict") or \
            selected_kwargs.get("cross_attention_aggregation_ids_dict"):
                raise ValueError(
                    "Standalone dit_decoder cannot use encoder aggregation; "
                    "use dit_encoder_decoder instead."
                )

            patch_size = selected_kwargs.get("patch_size", 2)

            # Infer the decoder's source grid from image and patch sizes.
            if selected_kwargs.get("encoder_output_grid_size") is None:
                selected_kwargs["encoder_output_grid_size"] = (
                    image_shape[0] // patch_size
                )

            # Default decoder source width to the transformer embedding width.
            if selected_kwargs.get("encoder_output_dim") is None:
                selected_kwargs["encoder_output_dim"] = selected_kwargs.get(
                    "dim", 32
                )

            # Enforce standalone-decoder conditioning settings in typed mode.
            if using_typed_model_config:
                selected_kwargs["decoder_separate_cond"] = True
                selected_kwargs["shift_inputs"] = False
                selected_kwargs["use_causal_mask"] = False
            # Provide safe standalone defaults without overriding direct choices.
            else:
                selected_kwargs.setdefault("decoder_separate_cond", True)
                selected_kwargs.setdefault("shift_inputs", False)
                selected_kwargs.setdefault("use_causal_mask", False)
            network = DiTDecoder(**selected_kwargs)
        # Construct the joint DiT encoder-decoder generator.
        elif name == "dit_encoder_decoder":
            decoder_kwargs = selected_kwargs.get("decoder_kwargs") or {}
            decoder_kwargs.setdefault("shift_inputs", False)
            selected_kwargs["decoder_kwargs"] = decoder_kwargs
            network = DiTEncoderDecoder(**selected_kwargs)
        # Construct the classified DiT encoder-decoder network.
        elif name == "dit_encoder_decoder_classifier":
            decoder_kwargs = selected_kwargs.get("decoder_kwargs") or {}
            decoder_kwargs.setdefault("shift_inputs", False)
            selected_kwargs["decoder_kwargs"] = decoder_kwargs
            network = DiTEncoderDecoderClassifier(**selected_kwargs)
        # Construct the generator-only U-Net.
        elif name == "unet":
            network = UNet(**selected_kwargs)
        # Construct the remaining supported U-Net classifier family.
        else:
            network = UNetClassifier(**selected_kwargs)

        selected_wrapper_name = wrapper_name
        selected_wrapper_kwargs = deepcopy(wrapper_kwargs)
        selected_wrapper_kwargs.setdefault(
            "test_steps", 
            min(50, network.timesteps)
        )
        # Select the wrapper implied by the raw network family.
        if selected_wrapper_name is None:
            selected_wrapper_name = "diffusion_classifier" if name in {
                "dit_classifier", 
                "dit_encoder_decoder_classifier", 
                "unet_classifier"
            } else "diffusion_model"
        # Normalize an explicitly requested wrapper name.
        else:
            selected_wrapper_name = str(selected_wrapper_name).lower()

        # Prevent classifier wrappers from receiving generator-only networks.
        if selected_wrapper_name in {
            "diffusion_classifier", "diffusion_classifier_v2"
        } and name not in {
            "dit_classifier",
            "dit_encoder_decoder_classifier",
            "unet_classifier",
        }:
            raise ValueError(
                f"Wrapper {selected_wrapper_name!r} requires a classifier network."
            )

        # Wrap the network with joint diffusion classification behavior.
        if selected_wrapper_name == "diffusion_classifier":
            selected_wrapper_kwargs.setdefault(
                "mask_by_nulls", 
                bool(network.use_cfg)
            )
            model = DiffusionClassifier(
                network=network, 
                **selected_wrapper_kwargs
            )
        # Wrap the network with separately optimized V2 classification behavior.
        elif selected_wrapper_name == "diffusion_classifier_v2":
            selected_wrapper_kwargs.setdefault(
                "mask_by_nulls", 
                bool(network.use_cfg)
            )
            model = DiffusionClassifierV2(
                network=network, 
                **selected_wrapper_kwargs
            )
        # Wrap a generator network with diffusion training and sampling.
        elif selected_wrapper_name == "diffusion_model":
            model = DiffusionModel(
                network=network, 
                **selected_wrapper_kwargs
            )
        # Reject wrapper names outside the three supported implementations.
        else:
            raise ValueError(
                "Unsupported model wrapper: " + str(selected_wrapper_name)
            )

        model.compile(**{
            "optimizer": optimizer, 
            "loss": loss_function, 
            **diffusion_compile_args
        })

        return model


    def finalize_selected(selected_model: Any) -> Any:
        """Optionally summarize and initialize a selected model from weights.

        Args:
            selected_model (tf.keras.Model): Newly constructed model.

        Returns:
            tf.keras.Model: The same model after optional weight loading.
        """
        # Build an uninitialized VAE before loading shape-dependent weights.
        if weights_path is not None \
        and isinstance(selected_model, VariationalAutoencoder) \
        and not getattr(selected_model, "built", False):
            import tensorflow as tf


            x = tf.zeros((1, flat_dim), dtype=tf.float32)
            inputs = (x, tf.one_hot([0], class_num)) if selected_model.conditioned else x
            selected_model(inputs, training=False)

        # Print the most useful available architecture summary when requested.
        if show_network_summary:
            # Summarize the wrapper directly once it is built.
            if getattr(selected_model, "built", False):
                selected_model.summary()
            # Fall back to built subcomponent summaries for lazy wrappers.
            else:
                summarized = False
                for attribute in (
                    "network", "encoder", 
                    "decoder", "classifier"
                ):
                    component = getattr(selected_model, attribute, None)
                    # Summarize each built component that exposes Keras summary.
                    if component is not None and getattr(component, "built", False) \
                    and hasattr(component, "summary"):
                        component.summary()
                        summarized = True

                # Let Keras report the wrapper when no component was available.
                if not summarized:
                    selected_model.summary()

        # Restore configured weights after the model has been initialized.
        if weights_path is not None:
            selected_model.load_weights(weights_path)

        return selected_model


    # Return the classifier/generator bundle required by continual learning.
    if task == "continual":
        # Return classifier-only continual bundles without a replay model.
        if model_name in _CLASSIFIER_MODELS:
            classifier = finalize_selected(build_selected(model_name))
            return {
                "classifier": classifier, 
                "classifier_name": model_name, 
                "generative_model": None
            }

        selected_classifier_name = classifier_name or (
            "dnn" if model_name in _VAE_MODELS else "cnn"
        )
        selected_classifier_name = selected_classifier_name.lower()
        # Keep continual pretrained classifiers on their required geometry.
        if pad and selected_classifier_name in {"pretrained", "hp-tuned"}:
            raise ValueError(
                "pad is not supported for pretrained/hp-tuned classifiers."
            )

        use_buffer = config.continually_learn.use_buffer \
                    if config is not None else kwargs.get("use_buffer", False)
        classifier = build_classifier(
            selected_classifier_name, 
            classifier_kwargs
        )
        generative_model = None if use_buffer else build_selected(model_name)
        # Apply configured weights to the classifier in buffer-only mode.
        if generative_model is None:
            classifier = finalize_selected(classifier)
        # Otherwise apply configured weights to the replay model.
        else:
            generative_model = finalize_selected(generative_model)

        return {
            "classifier": classifier, 
            "classifier_name": selected_classifier_name, 
            "generative_model": generative_model
        }

    return finalize_selected(build_selected(model_name))


def _classifier_name(name: object) -> str:
    """Return the legacy classifier spelling expected by its builder.

    Args:
        name (object): Classifier name convertible to text.

    Returns:
        str: Uppercase ``CNN``/``DNN`` or a lowercase remaining name.
    """

    name = str(name).lower()

    return name.upper() if name in ("cnn", "dnn") else name


def copy_model(prev_model: Any, new_model: Any) -> None: # , copy_opt_states=False
    """Copy a classifier while expanding its softmax head by one class.

    All non-final layers receive exact copies of their predecessors' weights.
    The old output weights and biases are copied into every column except the
    final column of ``new_model``; that last class retains its initializer.
    Optimizer state is not copied.

    Args:
        prev_model (tf.keras.Model): Built source classifier with ``L`` layers
            and final kernel shape ``[..., old_classes]``.
        new_model (tf.keras.Model): Built destination with the same ``L`` layer
            count and final width exactly ``old_classes + 1``.  Corresponding
            non-final layer weight shapes must match.

    Returns:
        None: ``new_model`` is modified in place.

    Raises:
        ValueError: If layer counts, corresponding weights, or output-head
            widths are incompatible.
    """
    # from tensorflow.keras import backend as K

    # import numpy as np


    layers_num = len(prev_model.layers)
    # Require matching layer structures before copying classifier weights.
    if layers_num != len(new_model.layers):
        raise ValueError("Source and destination models must have equal layer counts.")


    for i in range(layers_num-1):
        new_model.layers[i].set_weights(
            prev_model.layers[i].get_weights()
        )

    old_last_layer_weights, old_last_layer_bias = prev_model.layers[-1].get_weights()
    new_last_layer_weights, new_last_layer_bias = new_model.layers[-1].get_weights()

    new_last_layer_weights[..., :-1] = old_last_layer_weights
    new_last_layer_bias[:-1] = old_last_layer_bias

    new_model.layers[-1].set_weights([new_last_layer_weights, new_last_layer_bias])

    # if not copy_opt_states:
    #     return

    # lr = new_model.optimizer.learning_rate
    # K.set_value(new_model.optimizer.lr, 0.)
    # new_model.train_on_batch(
    #     np.random.normal(size=(1, *prev_model.input_shape[1:])),
    #     np.zeros((1,), dtype="float32")
    # )
    # K.set_value(new_model.optimizer.lr, lr)

    # prev_states = prev_model.optimizer.get_weights()
    # new_states = new_model.optimizer.get_weights()

    # new_states[:-2] = prev_states[:-2]
    # last_weight_prev_state, last_bias_prev_state = prev_states[-2:]

    # new_states[-2][:, :-1] = last_weight_prev_state
    # new_states[-1][:-1] = last_bias_prev_state

    # new_model.optimizer.set_weights(new_states)

    pass
