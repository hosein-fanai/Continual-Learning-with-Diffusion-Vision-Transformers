import os


models_path = "./models"
hyperas_path = os.path.join(models_path, "hyperas")

best_score = -float("inf")
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


def extract_features(dataset_list, batch_size=128, 
                    file_name=None):
    from tensorflow.keras import models

    import numpy as np


    conv_base = models.Sequential(
        get_model(10, model_type="pretrained", verbose=0).layers[:4]
    )
    conv_base.trainable = False

    features_list = []
    for dataset in dataset_list:
        features = conv_base.predict(dataset, batch_size=batch_size)
        features_list.append(features)

    del conv_base

    if file_name is not None:
        save_samples(np.array(features_list, dtype="object"), 
                    file_name, ".npy")

    return features_list


def CL_plot(class_num, pairs):
    from matplotlib import pyplot as plt


    for accs, label in pairs:
        plt.plot(list(range(2, class_num+1)), accs, label=label)

    plt.legend()
    plt.xlabel("#classes")
    plt.ylabel("accuracy")
    plt.show()


def plot_history(history, 
                range_=(0, None), 
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
        
        epochs = range(1, len(history.get(metric))+1)[range_]

        plt.figure(i)
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


def save_logs(model_name, i, search_space=[], 
            names=[], metrics={}, where_to="file"):
    txt = ""

    if search_space and names:
        txt += f"----Optimization Iteration {i}:\n"
        for ss, name in zip(search_space, names):
            txt += f"{name}: {ss}\n"

    if metrics:
        txt += "----("
        for metric_name, metric_value in metrics.items():
            txt += f"{metric_name}={metric_value}, "

        txt = txt[:-2] + ")\n\n"

    if where_to == "file" or where_to == "both":
        with open(f"./models/hyperas/logs/{model_name}.txt", "at") as f: 
            f.write(txt)

    if where_to == "print" or where_to == "both":
        print(txt)
