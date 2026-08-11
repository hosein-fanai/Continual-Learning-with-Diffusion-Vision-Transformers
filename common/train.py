import tensorflow as tf
from tensorflow.keras import optimizers, datasets, callbacks

import pandas as pd

import os

from common.utils import plot_images, plot_history, create_gif
from common.lr_logger_callback import LrLoggerCallback
from common.config import Config, load_config, save_config

from diffusion.models.transformer.diffusion_transformer import DiffusionTransformer
from diffusion.models.transformer.di_t_classifier import DiTClassifier
from diffusion.models.wrapper.diffusion_model import DiffusionModel
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from diffusion.callbacks.image_generator_callback import ImageGeneratorCallback


def get_datasets(config: Config):
    def get_dataset(x, y, 
                    shuffle_buffer=10_000, 
                    batch_size=128, 
                    drop_remainder=True):
        x = x.astype("float32") / 255.
        x = (x * 2.) - 1.
        x = x[..., None]

        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        if shuffle_buffer > 0:
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = dataset.batch(batch_size, drop_remainder)

        return dataset


    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    trainset = get_dataset(
        x_train, 
        y_train, 
        shuffle_buffer=config.dataset.shuffle_buffer, 
        batch_size=config.dataset.batch_size, 
        drop_remainder=True
    )
    outputs = [trainset]
    config.dataset.trainset_len = len(trainset)

    if config.training.use_valset:
        valset = get_dataset(
            x_test, 
            y_test, 
            shuffle_buffer=0, 
            batch_size=config.dataset.batch_size, 
            drop_remainder=False
        )
        outputs.append(valset)
    else:
        outputs.append(None)

    return outputs


def get_model(config: Config):
    if config.model.with_classifier:
        model = DiffusionClassifier(
            network=DiTClassifier(
                **config.model.dit_classifier.kwargs()
            ), 
            **config.model.diffusion_classifier.kwargs()
        )
    else:
        model = DiffusionModel(
            DiffusionTransformer(
                **config.model.diffusion_transformer.kwargs()
            ), 
            **config.model.diffusion_model.kwargs()
        )

    if config.model.show_network_summary:
        model.summary()

    if config.model.weights_path is not None:
        model.load_weights(config.model.weights_path)

    if config.optimizer.decay_steps is None:
        config.optimizer.decay_steps = config.training.epochs * config.dataset.trainset_len

    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=config.optimizer.initial_learning_rate, 
        decay_steps=config.optimizer.decay_steps, 
    )

    model.compile(
        optimizer=optimizers.Adam(lr_schedule), 
        loss="mse", 
    )

    return model


def train_model(
    config: Config, 
    model, 
    trainset, 
    valset=None, 
    save_config_=True, 
):
    callbacks_list = [
        LrLoggerCallback(), 
        callbacks.ProgbarLogger(count_mode="steps"), 
        ImageGeneratorCallback(
            show_images=config.training.show_images, 
            save_gifs=config.training.save_gifs, 
            results_path=config.training.results_path, 
            project_tag=config.training.project_tag
        ), 
    ]

    if save_config_:
        config_path = os.path.join(
            callbacks_list[2].results_path, 
            "config.yaml"
        )
        save_config(config, config_path)

    history = model.fit(
        trainset, 
        epochs=config.training.epochs, 
        validation_data=valset, 
        callbacks=callbacks_list, 
    ).history

    config.reporting.results_path = callbacks_list[-1].results_path

    if config.training.save_weights:
        config.model.weights_path = os.path.join(
            config.reporting.results_path, 
            "model.weights.h5"
        )
        model.save_weights(config.model.weights_path)

    if save_config_:
        save_config(config, config_path)

    return history


def report(
    config: Config, 
    history, 
    model, 
    trainset, 
    valset=None
):
    results_path = config.reporting.results_path

    plot_save_path = None
    plot_save_path_without_20percent = None
    if config.reporting.save_history_plot:
        plot_save_path = os.path.join(
            results_path, 
            "train history.png"
        )
        plot_save_path_without_20percent = os.path.join(
            results_path, 
            "train history without first 20percent.png"
        )
    csv_save_path = None
    if config.reporting.save_history_csv:
        csv_save_path = os.path.join(
            results_path, 
            "train history.csv"
        )

    plot_history(
        history, 
        show_plots=config.reporting.show_history_plot, 
        plot_path=plot_save_path, 
        csv_path=csv_save_path
    )
    if config.reporting.plot_without_20percent:
        plot_history(
            history, 
            range_=(int(0.2*config.training.epochs), None), 
            show_plots=config.reporting.show_history_plot, 
            plot_path=plot_save_path_without_20percent, 
        )

    eval_results = {}
    if config.reporting.run_trainset_eval:
        print("Trainset evaluation:")

        print("EMA Network:")
        trainset_ema_eval = model.evaluate(trainset, return_dict=True)

        print("Network:")
        trainset_network_eval = model.evaluate(trainset, network_name="", return_dict=True)

        eval_results["trainset_ema_eval"] = trainset_ema_eval
        eval_results["trainset_network_eval"] = trainset_network_eval

    if config.reporting.run_valset_eval:
        print("Valset evaluation:")

        print("EMA Network:")
        valset_ema_eval = model.evaluate(valset, return_dict=True)

        print("Network:")
        valset_network_eval = model.evaluate(valset, network_name="", return_dict=True)

        eval_results["valset_ema_eval"] = valset_ema_eval
        eval_results["valset_network_eval"] = valset_network_eval

    if len(eval_results) > 0 and config.reporting.save_evals_csv:
        eval_results_df = pd.DataFrame(eval_results).T
        eval_results_df.index.name = "dataset + network type"
        eval_results_df.to_csv(
            os.path.join(
                results_path, 
                "evals history.csv"
            ), 
            index=True
        )

    if config.reporting.save_final_gifs:
        imgs, frames1, frames2 = model.sample(
            labels=list(range(model.network.num_labels)), 
            scale=config.reporting.final_images_cfg_scale, 
            steps=config.reporting.final_images_steps, 
            return_x_ts=True, return_x0s=True
        )
        create_gif(
            os.path.join(
                results_path, 
                f"final-gifs_steps-{config.reporting.final_images_steps}_scale-{config.reporting.final_images_cfg_scale:.1f}.gif"
            ), 
            frames1, frames2
        )
    else:
        imgs = model.sample(
            labels=list(range(config.num_classes+1)), 
            scale=config.reporting.final_images_cfg_scale, 
            steps=config.reporting.final_images_steps
        )

    imgs_save_path = None
    if config.reporting.save_final_images:
        imgs_save_path = os.path.join(
            results_path, 
            f"final-images_steps-{config.reporting.final_images_steps}_scale-{config.reporting.final_images_cfg_scale:.1f}.png"
        )
    plot_images(
        imgs, 
        show_images=config.reporting.show_final_images, 
        save_path=imgs_save_path
    )


def main(config: Config):
    print("Initiating training process with the following settings:")
    print(config)


    trainset, valset = get_datasets(config)

    model = get_model(config)

    history = train_model(
        config, 
        model, 
        trainset, 
        valset=valset if config.training.use_valset else None, 
    )

    report(
        config, 
        history, 
        model, 
        trainset, 
        valset=valset
    )


if __name__ == "__main__":
    config = load_config()
    main(config)
