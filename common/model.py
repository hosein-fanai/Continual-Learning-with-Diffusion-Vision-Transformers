"""Legacy classifier factories, callbacks, and weight-expansion utilities."""


def get_compile_args(optimizer="adam", metrics=["accuracy"], 
                    loss="sparse_categorical_crossentropy"):
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


def get_callbacks(indices=[0], monitor="val_accuracy", mode="max", 
                patience=5, min_delta=1e-2, reducelr_factor=0.6, 
                verbose=1):
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
            verbose=verbose,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            factor=reducelr_factor,
            verbose=verbose,
        )
    ]

    return [callbacks_list[idx] for idx in indices]


def get_model(class_num, model_type="CNN", model_path="", 
            dropout_rate=0., num_last_not_frozen=3, 
            resize=(299, 299), compile_args={}, 
            use_loaded_opt=False, verbose=1):
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

    Returns:
        tf.keras.Sequential: A built, compiled classifier mapping a batch of
        images/features to ``float32`` probabilities shaped
        ``[batch, class_num]``.

    Raises:
        Exception: If ``model_type`` is not one of the four supported strings.

    Note:
        ``hp-tuned`` clones layer configuration, not the loaded layer weights;
        :func:`copy_model` is used separately when prior weights are required.
    """
    import tensorflow as tf
    from tensorflow.keras import models, layers, applications


    compile_args_default = get_compile_args()
    compile_args = {**compile_args_default, **compile_args}

    if model_type == "pretrained":
        conv_base = applications.Xception(include_top=False, input_shape=(resize[0], resize[1], 3))
        for layer in conv_base.layers[:-num_last_not_frozen]:
            layer.trainable = False

        model = models.Sequential([
            layers.Lambda(lambda X: tf.image.resize(X, resize), input_shape=(32, 32, 3), name="resize"),
            layers.Lambda(lambda X: applications.xception.preprocess_input(X), name="xception_preprocess"),
            conv_base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(class_num, activation="softmax")
        ])
    elif model_type == "hp-tuned":
        model = models.load_model(model_path)
        if use_loaded_opt:
            compile_args["optimizer"] = model.optimizer

        model = models.Sequential([
            *models.clone_model(model).layers[:-1],
            layers.Dense(class_num, activation="softmax")
        ])
    elif model_type == "CNN":
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
    elif model_type == "DNN":
        model = models.Sequential([
            # layers.Flatten(input_shape=(10, 10, 2048)),
            # layers.GlobalAveragePooling2D(input_shape=(10, 10, 2048)),
            # layers.Dense(256, activation="relu"),
            layers.Dropout(dropout_rate, input_shape=(2048,)),
            layers.Dense(class_num, activation="softmax")
        ])
    else:
        raise Exception("model_type needs to be one of pretrained, hp-tuned, CNN, or DNN.")

    model.compile(**compile_args)

    model.build(model.layers[0].input_shape)

    if verbose:
        model.summary()

    return model


def copy_model(prev_model, new_model): # , copy_opt_states=False
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
        AssertionError: If the models have different layer counts.
        ValueError: If corresponding weights are shape-incompatible or the new
            output layer is not exactly one class wider.
    """
    # from tensorflow.keras import backend as K

    # import numpy as np


    assert (layers_num:=len(prev_model.layers)) == len(new_model.layers)

    for i in range(layers_num-1):
        new_model.layers[i].set_weights(prev_model.layers[i].get_weights())

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
