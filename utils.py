import os


models_path = "./models"
hyperas_path = os.path.join(models_path, "hyperas")

best_acc = -float("inf")
i = 1


def init():
    import tensorflow as tf


    if gpus:=tf.config.list_physical_devices("GPU"):
        try:
            tf.config.set_logical_device_configuration(gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=6144)])
        except RuntimeError as e:
            print(e)
            print("Could not limit gpu memory.")


def sort_filter_labels(labels_list, indices):
    import numpy as np


    labels_set_list = []
    for labels_set in labels_list:
        allowed_instances_list = []
        for index in indices:
            allowed_instances = np.where(labels_set == index)[0].tolist()
            allowed_instances_list.extend(allowed_instances)

        labels_set_list.append(allowed_instances_list)
    
    return labels_set_list


def get_data(x_train, y_train, x_test, y_test, 
            indices, preprocess, return_features, 
            features_path, verbose):
    from sklearn.model_selection import train_test_split

    import numpy as np


    if preprocess:
        x_train = x_train.astype("float64") / 255.
        x_test = x_test.astype("float64") / 255.
    else:
        x_train = x_train.astype("uint8")
        x_test = x_test.astype("uint8")

    if return_features:
        x_train, x_val, x_test = load_samples(features_path, ".npy")

        labels_set_list = sort_filter_labels([y_train, y_test], indices)
        y_train = y_train[labels_set_list[0]]
        y_test = y_test[labels_set_list[1]]

        y_train, y_val = train_test_split(y_train, test_size=0.2, 
                                        stratify=y_train, random_state=42)

        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        labels_set_list = sort_filter_labels([y_train, y_test], indices)
        x_train, y_train = x_train[labels_set_list[0]], y_train[labels_set_list[0]]
        x_test, y_test = x_test[labels_set_list[1]], y_test[labels_set_list[1]]

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, 
                                                    stratify=y_train, random_state=42)

    if verbose:
        print("Trainset:", x_train.shape, y_train.shape)
        print("Validation set:", x_val.shape, y_val.shape)
        print("Testset:", x_test.shape, y_test.shape)

        for set_id, dataset in enumerate((y_train, y_val, y_test)):
            print(f"---{set_id}")

            for clss_id in np.unique(dataset):
                print(clss_id, sum(dataset == clss_id) / len(dataset))
            
            print()

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_cifar10(indices=list(range(10)), preprocess=True,  
                features_path="./data/cifar10_xception_gavgpooled_features_train_val_test", 
                return_features=False, verbose=1):
    from tensorflow.keras.datasets import cifar10


    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    return get_data(x_train, y_train, x_test, y_test, 
                indices, preprocess, return_features, 
                features_path, verbose)


def load_cifar100(indices=list(range(100)), preprocess=True, 
                features_path="./data/cifar100_xception_gavgpooled_features_train_val_test", 
                return_features=False, verbose=1):
    from tensorflow.keras.datasets import cifar100


    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    return get_data(x_train, y_train, x_test, y_test, 
                indices, preprocess, return_features, 
                features_path, verbose)


def get_dataset(X, Y, conv_base=None, batch_size=128):
    import tensorflow as tf


    def preprocess_func(x, y):
        if conv_base is not None:
            x = conv_base(x)

        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_func)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.cache()

    return dataset


def create_compile_args(optimizer="adam"):
    compile_args = {
        "optimizer": optimizer,
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"]
    }

    return compile_args


def create_callbacks_list(monitor="val_accuracy", mode="max", 
                        patience=5, min_delta=1e-3, reducelr_factor=0.6, 
                        idx=[0], verbose=1):
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
    
    return [callbacks_list[i] for i in idx]


def get_callbacks():
    from tensorflow.keras import callbacks


    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            restore_best_weights=True,
            patience=5,
            min_delta=1e-2,
            verbose=1,
        )
    ]


def get_model(class_num, model_type="CNN", model_path="", 
            dropout_rate=0., num_last_not_frozen=3, 
            resize=(299, 299), verbose=1):
    import tensorflow as tf
    from tensorflow.keras import models, layers, applications


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
        model = models.Sequential([
            *models.clone_model(models.load_model(model_path)).layers[:-1],
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

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.build(model.layers[0].input_shape)

    if verbose:
        model.summary()

    return model


def copy_model(prev_model, new_model):
    from tensorflow.keras import layers


    for old_layer, new_layer in zip(prev_model.layers, new_model.layers):
        if isinstance(old_layer, layers.Dense):
            break

        new_layer.set_weights(old_layer.get_weights())

    old_last_layer_weights, old_last_layer_bias = prev_model.layers[-1].get_weights()
    new_last_layer_weights, new_last_layer_bias = new_model.layers[-1].get_weights()

    new_last_layer_weights[..., :-1] = old_last_layer_weights
    new_last_layer_bias[:-1] = old_last_layer_bias

    new_model.layers[-1].set_weights([new_last_layer_weights, new_last_layer_bias])


def plot_history(history, range_=(0, None), 
                indices=None):
    from matplotlib import pyplot as plt


    range_ = slice(*range_)
    metrics = list(history.keys())
    has_val = history.get("val_"+metrics[0])

    if indices is None:
        indices = list(range(len(metrics)))

    for i, metric in enumerate(metrics):
        i += 1

        if i-1 not in indices:
            continue

        if (i > len(history.keys()) / 2 and has_val):
            break

        plt.figure(i)
        
        epochs = range(1, len(history.get(metric))+1)[range_]
        plt.plot(epochs, history.get(metric)[range_], label="Training")
        if has_val:
            plt.plot(epochs, history.get("val_"+metric)[range_], label="Validation", marker='v')
    
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.show()


def show_img(x, y=None):
    from matplotlib import pyplot as plt


    plt.imshow(x)
    plt.axis("off")

    if y[0] is not None:
        plt.title(f"Label: {y[0]}")

    plt.show()


def save_samples(arr, path, type_):
    import numpy as np


    if type_ == ".csv":
        np.savetxt(path+type_, arr, delimiter=',')
    elif type_ == ".npy":
        with open(path+type_, "wb") as file:
            np.save(file ,arr)
    else:
        print("Wrong type!")


def load_samples(path, type_):
    import numpy as np


    if type_ == ".csv":
        pass
    elif type_ == ".npy":
        with open(path+type_, "rb") as file:
            arr = np.load(file, allow_pickle=True)
    else:
        return None
    
    return arr


def save_logs(model_name, i, val_f1, best_f1, 
            search_space, names, where_to="file"):
    txt = f"----Opt Iteration {i}: val_acc={val_f1}, best_acc={best_f1}\n"
    for ss, name in zip(search_space, names):
        txt += f"{name}: {ss}\n"
    txt += '\n'

    if where_to == "file" or where_to == "both":
        with open(f"./models/hyperas/logs/{model_name}.txt", "at") as f: 
            f.write(txt)

    if where_to == "print" or where_to == "both":
        print(txt)


