def get_compile_args(optimizer="adam", metrics=["accuracy"], 
                    loss="sparse_categorical_crossentropy"):
    compile_args = {
        "optimizer": optimizer,
        "loss": loss,
        "metrics": metrics
    }

    return compile_args


def get_callbacks(indices=[0], monitor="val_accuracy", mode="max", 
                patience=5, min_delta=1e-2, reducelr_factor=0.6, 
                verbose=1):
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
