"""End-to-end MNIST construction, training, and reporting orchestration.

Raw networks from ``diffusion.models.transformer`` implement one neural
forward pass.  Wrappers from ``diffusion.models.wrapper`` own the diffusion
schedule, EMA copy, noising, custom optimization steps, evaluation, and
iterative sampling around the selected network.
"""

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


def get_datasets(config: Config | None = None):
    """Load MNIST and construct normalized training/validation pipelines.

    Args:
        config (Config | None): Uses the configured dataset settings when
            provided.  ``None`` uses the existing MNIST defaults.

    Returns:
        list[tf.data.Dataset | None]: ``[trainset, valset]``.  Training batches
        drop the final remainder and are shuffled when the buffer is positive.
        The validation set is the MNIST test split with its last partial batch
        retained, or ``None`` when validation is disabled.  Images are
        ``float32`` tensors shaped ``[batch, 28, 28, 1]`` in ``[-1, 1]``;
        labels are integer tensors shaped ``[batch]``.
    """

    batch_size = 128 if config is None else config.dataset.batch_size
    shuffle_buffer = 10_000 if config is None else config.dataset.shuffle_buffer
    use_valset = True if config is None else config.training.use_valset


    def get_dataset(x, y, 
                    shuffle_buffer=10_000, 
                    batch_size=128, 
                    drop_remainder=True):
        """Normalize arrays and create one batched ``tf.data`` pipeline.

        Args:
            x (numpy.ndarray): Grayscale images shaped ``[samples, height,
                width]`` with values in ``[0, 255]``.
            y (numpy.ndarray): Aligned integer labels shaped ``[samples]``.
            shuffle_buffer (int): Positive shuffle capacity; ``0`` or a
                negative value preserves input order.
            batch_size (int): Positive examples per batch.
            drop_remainder (bool): Whether to omit a final undersized batch.

        Returns:
            tf.data.Dataset: Batched ``(images, labels)`` pairs.  Images have a
            newly appended channel axis and values linearly scaled to
            ``[-1, 1]``.
        """
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
        shuffle_buffer=shuffle_buffer,
        batch_size=batch_size,
        drop_remainder=True
    )
    outputs = [trainset]
    if config is not None:
        config.dataset.trainset_len = len(trainset)

    if use_valset:
        valset = get_dataset(
            x_test, 
            y_test, 
            shuffle_buffer=0, 
            batch_size=batch_size, 
            drop_remainder=False
        )
        outputs.append(valset)
    else:
        outputs.append(None)

    return outputs


def get_model(
    config: Config | None = None, 
    *, 
    trainset_len=None, 
    epochs=20, 
):
    """Build and compile the configured raw network and diffusion wrapper.

    With classification enabled, ``DiTClassifier`` is wrapped by
    ``DiffusionClassifier``.  Otherwise ``DiffusionTransformer`` is wrapped by
    ``DiffusionModel``.  Leaf config dictionaries are expanded directly into
    their constructors; no legacy-key translation is performed.

    Args:
        config (Config | None): Model and optimizer settings.  ``None`` builds
            the default classifier model.
        trainset_len (int | None): Number of training batches in ``None`` mode.
        epochs (int): Epoch count used to derive decay steps in ``None`` mode.

    Returns:
        diffusion.models.wrapper.diffusion_model.DiffusionModel: A compiled
        wrapper (possibly the ``DiffusionClassifier`` subclass) using Adam,
        cosine learning-rate decay, and mean-squared-error compiled loss.

    Raises:
        AttributeError: If automatic decay is requested before
            ``config.dataset.trainset_len`` is set.
        ValueError: If ``trainset_len`` is omitted in ``None`` mode.
        OSError: If ``weights_path`` is set but cannot be loaded.

    Note:
        When ``decay_steps`` is ``None``, this function mutates it to
        ``training.epochs * dataset.trainset_len``.
    """

    if config is None:
        model = DiffusionClassifier(
            network=DiTClassifier()
        )
        show_network_summary = True
        weights_path = None
        initial_learning_rate = 5e-3

        if trainset_len is None:
            raise ValueError("trainset_len is required when config is None.")

        decay_steps = epochs * trainset_len
    elif config.model.with_classifier:
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

    if config is not None:
        show_network_summary = config.model.show_network_summary
        weights_path = config.model.weights_path
        initial_learning_rate = config.optimizer.initial_learning_rate

        if config.optimizer.decay_steps is None:
            config.optimizer.decay_steps = (
                config.training.epochs * config.dataset.trainset_len
            )
        decay_steps = config.optimizer.decay_steps

    if show_network_summary:
        model.summary()

    if weights_path is not None:
        model.load_weights(weights_path)

    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate, 
        decay_steps=decay_steps, 
    )

    model.compile(
        optimizer=optimizers.Adam(lr_schedule), 
        loss="mse", 
    )

    return model


def train_model(
    config: Config | None, 
    model, 
    trainset, 
    valset=None, 
    save_config_=True, 
    *, 
    epochs=20, 
):
    """Fit a diffusion wrapper, manage callbacks, and persist run state.

    Args:
        config (Config | None): Supplies training settings when provided.
            ``None`` uses a display-only image callback and ``epochs``.
        model (tf.keras.Model): Compiled diffusion wrapper implementing Keras
            ``fit`` and ``save_weights``.
        trainset (tf.data.Dataset): Batched training ``(images, labels)`` pairs.
        valset (tf.data.Dataset | None): Optional validation pairs.
        save_config_ (bool): Save ``config.yaml`` before training and again
            after paths are updated.  It does not disable callback artifacts or
            weight saving.
        epochs (int): Fit epoch count used only when ``config`` is ``None``.

    Returns:
        dict[str, list[float]]: The Keras ``History.history`` mapping with one
        value per completed epoch for each reported metric.

    Side Effects:
        Sets ``config.reporting.results_path`` from
        ``ImageGeneratorCallback.results_path``.  If weight saving is enabled,
        sets ``config.model.weights_path`` and writes ``model.weights.h5``.
        Epoch trajectory sampling is skipped for no-EMA and VAE/swap wrappers.
    """

    if config is None:
        image_callback = ImageGeneratorCallback()
    else:
        image_callback = ImageGeneratorCallback(
            show_images=config.training.show_images, 
            save_gifs=config.training.save_gifs, 
            results_path=config.training.results_path, 
            project_tag=config.training.project_tag
        )

    callbacks_list = [
        LrLoggerCallback(), 
        callbacks.ProgbarLogger(count_mode="steps"), 
    ]
    if not isinstance(model, DiffusionModel) or (
        model.use_ema and not model.swap_noise_image
    ):
        callbacks_list.append(image_callback)

    if save_config_ and config is not None:
        config_path = os.path.join(
            image_callback.results_path, 
            "config.yaml"
        )
        save_config(config, config_path)

    history = model.fit(
        trainset, 
        epochs=epochs if config is None else config.training.epochs, 
        validation_data=valset, 
        callbacks=callbacks_list, 
    ).history

    if config is None:
        return history

    config.reporting.results_path = image_callback.results_path

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
    config: Config | None, 
    history, 
    model, 
    trainset, 
    valset=None
):
    """Create configured history, evaluation, sample-image, and GIF reports.

    Args:
        config (Config | None): Reporting settings and result path.  ``None``
            displays the history and final samples without writing artifacts.
        history (Mapping[str, Sequence[float]]): Epoch metric series, normally
            the return value of :func:`train_model`.
        model (DiffusionModel): Wrapper implementing ``evaluate`` with optional
            ``network_name`` and ``sample`` with CFG/trajectory controls.
        trainset (tf.data.Dataset): Batched training data used for optional raw
            and EMA evaluation.
        valset (tf.data.Dataset | None): Validation data.  It must be non-``None``
            when ``run_valset_eval=True``.

    Returns:
        None: Requested artifacts are displayed and/or written beneath the
        result path.  VAE/swap mode reports its final images without a GIF.

    Raises:
        AttributeError: If configured mode is called before a result path is
            assigned.
        ValueError: If requested sampling steps/scales or datasets violate the
            wrapper's constraints.
    """

    if config is None:
        plot_history(history)

        if trainset is not None:
            if getattr(model, "use_ema", True):
                print("Trainset evaluation (EMA Network):")
                model.evaluate(trainset, network_name="ema", return_dict=True)
            print("Trainset evaluation (Raw Network):")
            model.evaluate(trainset, network_name="raw", return_dict=True)

        if valset is not None:
            if getattr(model, "use_ema", True):
                print("Valset evaluation (EMA Network):")
                model.evaluate(valset, network_name="ema", return_dict=True)
            print("Valset evaluation (Raw Network):")
            model.evaluate(valset, network_name="raw", return_dict=True)

        network_name = "ema" if getattr(model, "use_ema", True) else "raw"
        imgs = model.sample(
            network_name=network_name, 
            labels=list(range(model.network.num_labels)), 
        )
        plot_images(imgs)

        return

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

        if model.use_ema:
            print("EMA Network:")
            trainset_ema_eval = model.evaluate(
                trainset, network_name="ema", return_dict=True
            )
            eval_results["trainset_ema_eval"] = trainset_ema_eval

        print("Raw Network:")
        trainset_network_eval = model.evaluate(
            trainset, network_name="raw", return_dict=True
        )

        eval_results["trainset_network_eval"] = trainset_network_eval

    if config.reporting.run_valset_eval:
        print("Valset evaluation:")

        if model.use_ema:
            print("EMA Network:")
            valset_ema_eval = model.evaluate(
                valset, network_name="ema", return_dict=True
            )
            eval_results["valset_ema_eval"] = valset_ema_eval

        print("Raw Network:")
        valset_network_eval = model.evaluate(
            valset, network_name="raw", return_dict=True
        )

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

    network_name = "ema" if model.use_ema else "raw"
    if config.reporting.save_final_gifs and not model.swap_noise_image:
        imgs, frames1, frames2 = model.sample(
            network_name=network_name, 
            labels=list(range(model.network.num_labels)), 
            scale=config.reporting.final_images_cfg_scale, 
            steps=config.reporting.final_images_steps, 
            return_x_ts=True, return_x0s=True
        )
        create_gif(
            os.path.join(
                results_path, 
                f"final-gifs_steps-{config.reporting.final_images_steps}"
                f"_scale-{config.reporting.final_images_cfg_scale:.1f}.gif"
            ), 
            frames1, frames2
        )
    else:
        imgs = model.sample(
            network_name=network_name, 
            labels=list(range(model.network.num_labels)), 
            scale=config.reporting.final_images_cfg_scale, 
            steps=config.reporting.final_images_steps
        )

    imgs_save_path = None
    if config.reporting.save_final_images:
        imgs_save_path = os.path.join(
            results_path, 
            f"final-images_steps-{config.reporting.final_images_steps}"
            f"_scale-{config.reporting.final_images_cfg_scale:.1f}.png"
        )
    plot_images(
        imgs, 
        show_images=config.reporting.show_final_images, 
        save_path=imgs_save_path
    )


def main(config: Config | None = None, *, epochs=20):
    """Run dataset setup, model construction, fitting, and final reporting.

    Args:
        config (Config | None): Complete training configuration.  ``None``
            runs the same pipeline with function and class defaults.
        epochs (int): Epoch count used only when ``config`` is ``None``.

    Returns:
        None: Progress is printed and configured artifacts are produced.
    """

    print("Initiating training process with the following settings:")
    print(config)

    trainset, valset = get_datasets(config)

    model = get_model(
        config, 
        trainset_len=len(trainset), 
        epochs=epochs, 
    )

    history = train_model(
        config, 
        model, 
        trainset, 
        valset=valset, 
        epochs=epochs, 
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
