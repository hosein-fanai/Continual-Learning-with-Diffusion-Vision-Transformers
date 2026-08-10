import os


models_path = "./models"
hyperas_path = os.path.join(models_path, "hyperas")

best_score = -float("inf")
i = 1


def init():
    import tensorflow as tf


    if gpus:=tf.config.list_physical_devices("GPU"):
        try:
            tf.config.set_logical_device_configuration(gpus[0], [
                tf.config.LogicalDeviceConfiguration(memory_limit=6144)
            ])
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


def plot_history(
    history, 
    range_=(0, None), 
    metrics=None, 
    row=None, col=3, 
    figsize=None, 
    x_ticks_rotation=90, 
    y_ticks_rotation=0, 
    show_all_x_ticks=True, 
    y_ticks_num=None, 
    show_plots=True, 
    plot_path=None, 
    csv_path=None
):
    from matplotlib import pyplot as plt

    import numpy as np

    import pandas as pd


    range_ = slice(*range_)

    if metrics is None:
        metrics = list(history.keys())

    plotted_metrics = []
    for metric in metrics:
        if metric in plotted_metrics:
            continue

        if metric.startswith("val_") and \
            metric.replace("val_", '') in metrics:
                continue

        plotted_metrics.append(metric)

    if row is None:
        row = -(len(plotted_metrics) // -col)

    if figsize is None:
        figsize = (20, row*5)

    fig, axes = plt.subplots(row, col, figsize=figsize)
    axes = axes.flatten()

    for i, metric in enumerate(plotted_metrics):
        ax = axes[i]
        epochs = range(1, len(history.get(metric))+1)[range_]
        show_all_x_ticks = False if len(epochs) > 50 else show_all_x_ticks

        values = history.get(metric)[range_]
        min_ = min(values)
        max_ = max(values)

        ax.plot(epochs, values, 
            label="Training" if not metric.startswith("val_") else "Validation")

        if values:=history.get("val_"+metric, None):
            min_ = min([min_]+values)
            max_ = max([max_]+values)

            ax.plot(epochs, values[range_], label="Validation")            

        ax.legend()
        ax.set_xlabel("epochs")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.tick_params(axis='x', rotation=x_ticks_rotation)
        ax.tick_params(axis='y', rotation=y_ticks_rotation)

        if show_all_x_ticks:
            ax.set_xticks(epochs)

        if y_ticks_num:
            ax.set_yticks(np.linspace(min_, max_, y_ticks_num))    

    for i in range(len(plotted_metrics), row*col):
        axes[i].set_visible(False)
    
    plt.tight_layout()

    if plot_path:
        fig.savefig(
            plot_path, 
            dpi=1_000, 
            bbox_inches="tight"
        )

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    if csv_path:
        history_df = pd.DataFrame(history)
        history_df.insert(0, "epoch", range(1, len(history_df) + 1))
        history_df.to_csv(csv_path, index=False)


def create_gif(
    output_path, 
    images1, 
    images2=None, 
    duration=100, 
    loop=0, 
    verbose=1
):
    import numpy as np
    
    from PIL import Image


    if images2 is None:
        images = images1
    else:
        images = []
        for image1, image2 in zip(images1, images2):
            images.append(
                np.concatenate([
                    image1, 
                    np.ones((image1.shape[0], 10, image1.shape[2], image1.shape[3])), 
                    image2
                ], axis=1)
            )

    frames = []
    for image in images:
        img = (image*255).astype("uint8")[..., 0]
        img = np.concatenate(img, axis=1)
        img = Image.fromarray(img)

        frames.append(img.convert("RGBA"))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )

    if verbose:
        print(f"GIF saved to {output_path}")


def show_img(x, y=None):
    from matplotlib import pyplot as plt


    plt.imshow(x)
    plt.axis("off")

    if y[0] is not None:
        plt.title(f"Label: {y[0]}")

    plt.show()


def plot_images(
    imgs, 
    row=1, col=11, 
    show_images=True, 
    save_path=None
):
    from matplotlib import pyplot as plt


    assert show_images or save_path is not None


    col = len(imgs) # // row
    fig, axes = plt.subplots(row, col, figsize=(20, 6))
    axes = axes.flatten()

    for i in range(len(imgs)):
        axes[i].imshow(imgs[i, :, :, 0], cmap="gray")
        axes[i].set_title(f"{i-1}") 
        axes[i].axis("off")

    for j in range(len(imgs), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(
            save_path, 
            dpi=1_000, 
            bbox_inches="tight", 
        )

    if show_images:
        plt.show()
    else:
        plt.close(fig)


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
