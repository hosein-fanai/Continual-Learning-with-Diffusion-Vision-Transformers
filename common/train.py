"""Config-driven dataset, model, training, and reporting orchestration.

Raw networks from ``diffusion.models.transformer`` implement one neural
forward pass.  Wrappers from ``diffusion.models.wrapper`` own the diffusion
schedule, EMA copy, noising, custom optimization steps, evaluation, and
iterative sampling around the selected network.
"""

import tensorflow as tf
from tensorflow.keras import utils, callbacks

import pandas as pd

import numpy as np

import os

import json

from copy import deepcopy
from collections.abc import Callable, Sequence

from common.utils import plot_images, plot_history, create_gif
from common.lr_logger_callback import LrLoggerCallback
from common.config import Config, load_config, save_config
from common.dataloader import get_datasets, get_dataset_spec
from common.model import get_model
from common.learner import _continually_learn

from autoencoder.variational_autoencoder import VariationalAutoencoder

from diffusion.models.wrapper.diffusion_model import DiffusionModel
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2
from diffusion.callbacks.image_generator_callback import ImageGeneratorCallback


def train_model(
    config: Config | None = None, 
    model: tf.keras.Model | dict[str, object] | None = None, 
    trainset: tf.data.Dataset | Callable[..., object] | None = None, 
    valset: tf.data.Dataset | None = None, 
    save_config_: bool = True, 
    extra_callbacks: Sequence[tf.keras.callbacks.Callback] | None = None, 
    **kwargs: object
) -> dict[str, list[float]]:
    """Fit a configured model, manage callbacks, and persist run state.

    Args:
        config (Config | None): Supplies training settings when provided.
            ``None`` uses a display-only image callback and ``epochs``.
        model (tf.keras.Model | dict): Compiled model or continual target/replay
            mapping returned by :func:`get_model`.
        trainset (tf.data.Dataset | Callable): Batched training pairs, or the
            CIFAR loader used by continual learning.
        valset (tf.data.Dataset | None): Optional validation pairs.
        save_config_ (bool): Save ``config.yaml`` before training and again
            after paths are updated.  It does not disable callback artifacts or
            weight saving.
        extra_callbacks (Sequence[tf.keras.callbacks.Callback] | None): Extra
            callbacks appended after the standard callbacks. HPO uses this for
            pruning; the continual learner uses it for per-task callbacks.
        **kwargs (object): Direct training, reporting, dataset-loader, and
            persistence settings used only when ``config`` is ``None``.
            Important fit keys are ``epochs``, ``verbose``, ``patience``,
            ``monitor``, ``monitor_mode``, ``report_every_epoch``,
            ``save_weights``, and callback/artifact settings. ``fit_method``
            selects an alternate model method such as VAE ``"train"`` or V2
            ``"fit_generator"``/``"fit_discriminator"``; ``fit_kwargs`` is
            shallow-copied and forwarded to that method. A continual model
            bundle consumes ``continual_kwargs`` spelling remains accepted in 
            direct mode) and also accepts ``max_train_samples``, 
            ``max_val_samples``, ``shuffle_buffer``, raw-image ``pad``, and 
            ``seed``.

    Returns:
        dict[str, list[float]]: The Keras ``History.history`` mapping with one
        value per completed epoch for each reported metric. Continual bundles
        return ``continual_accuracy`` and ``task_val_accuracy``.

    Side Effects:
        Sets ``config.training.results_path`` from
        ``ImageGeneratorCallback.results_path``. If weight saving is enabled,
        writes ``model.weights.h5`` and sets ``config.model.weights_path`` to
        the selected model's weights. A continual replay model is saved as
        ``replay-model.weights.h5`` and is the selected model in that case.
        TensorBoard and HPO metadata are written when configured. Epoch
        trajectory sampling is limited to compatible diffusion wrappers. A
        continual model bundle is updated with its final classifier/replay
        model and a ``continual_details`` mapping.
    """

    if config is None:
        show_images = kwargs.get("show_images", True)
        save_gifs = kwargs.get("save_gifs", False)
        results_path = kwargs.get("results_path", "./results")
        project_tag = kwargs.get("project_tag", "")
        report_every_epoch = kwargs.get("report_every_epoch", True)
        patience = kwargs.get("patience", 0)
        monitor = kwargs.get("monitor")
        monitor_mode = kwargs.get("monitor_mode", "auto")
        use_tensorboard = kwargs.get("tensorboard", False)
        tensorboard_run_name = kwargs.get("tensorboard_run_name", None)
        tensorboard_path = kwargs.get("tensorboard_path", None)
        hpo = kwargs.get("hpo", {})
        continual_kwargs = deepcopy(kwargs.get(
            "continual_kwargs", {}
        ))
        dataset_name = kwargs.get("dataset_name", "mnist")
        loader_preprocess = kwargs.get("preprocess", "standardize")
        features_path = kwargs.get("features_path", "")
        onehot_labels = kwargs.get("onehot_labels", False)
        validation_ratio = kwargs.get("validation_ratio", 0.)
        use_valset = kwargs.get("use_valset", True)
        seed = kwargs.get("seed")
        training_verbose = kwargs.get("verbose", 1)
        epochs = kwargs.get("epochs", 10)
        batch_size = kwargs.get("batch_size", 128)
        continual_return_features = kwargs.get("return_features")
        continual_max_train_samples = kwargs.get("max_train_samples")
        continual_max_val_samples = kwargs.get("max_val_samples")
        continual_shuffle_buffer = kwargs.get("shuffle_buffer")
        continual_pad = kwargs.get("pad", 0)
        initial_classifier = kwargs.get("initial_classifier")
        save_weights = kwargs.get("save_weights", False)
        classifier_compile_overrides = deepcopy(
            kwargs.get("classifier_kwargs", {}).get("compile_args", {})
        )
        fit_method = kwargs.get("fit_method", "fit")
        fit_kwargs = dict(kwargs.get("fit_kwargs", {}))
    else:
        show_images = config.training.show_images
        save_gifs = config.training.save_gifs
        results_path = config.training.results_path
        project_tag = config.training.project_tag
        report_every_epoch = config.training.report_every_epoch
        patience = config.training.patience
        monitor = config.training.monitor
        monitor_mode = config.training.monitor_mode
        use_tensorboard = config.training.tensorboard
        tensorboard_run_name = config.training.tensorboard_run_name
        tensorboard_path = config.training.tensorboard_path
        hpo = config.hpo
        continual_kwargs = config.continually_learn.kwargs()
        dataset_name = config.dataset.name
        loader_preprocess = config.dataset.preprocess
        features_path = config.dataset.features_path
        onehot_labels = config.dataset.onehot_labels
        validation_ratio = config.dataset.validation_ratio
        use_valset = config.training.use_valset
        seed = config.training.seed
        training_verbose = config.training.verbose
        epochs = config.training.epochs
        batch_size = config.dataset.batch_size
        continual_return_features = config.dataset.return_features
        continual_max_train_samples = config.dataset.max_train_samples
        continual_max_val_samples = config.dataset.max_val_samples
        continual_shuffle_buffer = config.dataset.shuffle_buffer
        continual_pad = config.dataset.pad
        initial_classifier = model.get("classifier") if isinstance(model, dict) \
                            and model.get("generative_model") is None \
                            and config.model.weights_path is not None else None
        save_weights = config.training.save_weights
        classifier_compile_overrides = deepcopy(config.model.classifier_kwargs.get(
            "compile_args", {}
        ))
        fit_method = "fit"
        fit_kwargs = {}

    callbacks_list = [
        LrLoggerCallback(), 
        callbacks.ProgbarLogger(count_mode="steps"), 
    ]

    image_callback = ImageGeneratorCallback(
        show_images=show_images, 
        save_gifs=save_gifs, 
        results_path=results_path, 
        project_tag=project_tag
    )
    if isinstance(model, DiffusionModel) and \
    report_every_epoch and not model.swap_noise_image:
        callbacks_list.append(image_callback)

    if not isinstance(model, dict) and  patience > 0:
        callbacks_list.append(callbacks.EarlyStopping(
            monitor=monitor, 
            mode=monitor_mode, 
            patience=patience, 
            restore_best_weights=True
        ))

    if config is not None:
        config.training.results_path = image_callback.results_path

    if use_tensorboard:
        tensorboard_name = tensorboard_run_name or "run"
        tensorboard_root = tensorboard_path or os.path.join(
            image_callback.results_path, 
            "tensorboard"
        )
        tensorboard_path = os.path.join(
            tensorboard_root, 
            project_tag or "run"
        )

        callbacks_list.append(callbacks.TensorBoard(
            log_dir=tensorboard_path, 
            histogram_freq=0, 
            write_graph=False
        ))

        writer = tf.summary.create_file_writer(
            tensorboard_path, 
            filename_suffix="." + tensorboard_name
        )
        with writer.as_default():
            tf.summary.text(
                "hyperparameters", 
                json.dumps(hpo.get("params", {}), sort_keys=True), 
                step=0
            )
        writer.flush()

    if extra_callbacks is not None:
        callbacks_list += list(extra_callbacks)

    if save_config_ and config is not None:
        config_path = os.path.join(
            image_callback.results_path, 
            "config.yaml"
        )
        save_config(config, config_path)

    if isinstance(model, dict):
        classifier_name = model["classifier_name"]
        template_path = os.path.join(
            image_callback.results_path, 
            classifier_name + "-template.h5"
        )
        model["classifier"].save(template_path)

        continual_kwargs = deepcopy(continual_kwargs)
        dataset_class_num, _, _ = get_dataset_spec(dataset_name)
        configured_class_num = continual_kwargs.pop("class_num", None)
        class_num = dataset_class_num if configured_class_num is None \
                    else configured_class_num
        if class_num < 2:
            raise ValueError("continual class_num must be at least 2.")

        generative_kwargs = continual_kwargs.pop(
            "generative_model_kwargs", {}
        )
        use_buffer = continual_kwargs.get("use_buffer", False)
        use_loaded_opt = continual_kwargs.pop("use_loaded_opt", True)
        continual_kwargs.pop("return_details", None)

        for factory_owned_key in (
            "load_dataset_fn", 
            "load_dataset_fn_kwargs", 
            "tuned_model_path", 
            "batch_size", 
            "epochs", 
            "generative_model", 
            "callbacks_list", 
            "verbose", 
            "return_features", 
            "max_train_samples", 
            "max_val_samples", 
            "shuffle_buffer", 
            "pad", 
            "dataset_seed", 
            "initial_classifier", 
            "callback_patience", 
            "callback_monitor", 
            "callback_monitor_mode"
        ):
            continual_kwargs.pop(factory_owned_key, None)

        classifier_compile_args = {
            "loss": "categorical_crossentropy" if onehot_labels else
                    "sparse_categorical_crossentropy", 
            "metrics": ["accuracy"], 
            **classifier_compile_overrides
        }
        continual_kwargs.setdefault(
            "compile_args", 
            classifier_compile_args
        )
        continual_kwargs.setdefault(
            "use_valset", 
            use_valset
        )

        loader_kwargs = {
            "preprocess": loader_preprocess, 
            "onehot_labels": onehot_labels, 
            "validation_ratio": validation_ratio, 
            "features_path": features_path, 
            "seed": seed
        }

        details = _continually_learn(
            class_num=class_num, 
            load_dataset_fn=trainset, 
            load_dataset_fn_kwargs=loader_kwargs, 
            tuned_model_path=template_path, 
            use_loaded_opt=use_loaded_opt, 
            batch_size=batch_size, 
            epochs=epochs, 
            generative_model=None if use_buffer else model["generative_model"], 
            generative_model_kwargs=generative_kwargs, 
            callbacks_list=callbacks_list[2:], 
            return_details=True, 
            return_features=continual_return_features, 
            max_train_samples=continual_max_train_samples, 
            max_val_samples=continual_max_val_samples, 
            shuffle_buffer=continual_shuffle_buffer, 
            pad=continual_pad, 
            dataset_seed=seed, 
            initial_classifier=initial_classifier, 
            callback_patience=patience, 
            callback_monitor=monitor, 
            callback_monitor_mode=monitor_mode, 
            verbose=training_verbose, 
            **continual_kwargs
        )

        model["classifier"] = details["model"]
        model["generative_model"] = details["generative_model"]
        model["continual_details"] = details
        history = {"continual_accuracy": details["accuracies"]}
        final_val_accuracy = [
            task_history.get(
                "val_accuracy", 
                task_history.get("val_classifier_accuracy", [np.nan])
            )[-1]
            for task_history in details["histories"]
        ]
        history["task_val_accuracy"] = final_val_accuracy
    elif fit_method != "fit":
        method = getattr(model, fit_method)

        if fit_method == "train":
            method_kwargs = {
                "epochs": epochs, 
                "validation_data": valset, 
                "callbacks_list": callbacks_list, 
                "verbose": training_verbose, 
                **fit_kwargs
            }
            trained = method(
                trainset, 
                **method_kwargs
            )
        else:
            method_kwargs = {
                "x": trainset, 
                "epochs": epochs, 
                "validation_data": valset, 
                "callbacks": callbacks_list, 
                "verbose": training_verbose, 
                **fit_kwargs
            }
            trained = method(
                **method_kwargs
            )

        history = getattr(trained, "history", trained)
    elif isinstance(model, DiffusionClassifierV2):
        fit_kwargs = {
            "x": trainset, 
            "epochs": epochs, 
            "validation_data": valset, 
            "callbacks": callbacks_list, 
            "verbose": training_verbose
        }
        history = model.fit(
            gen_kwargs=fit_kwargs, 
            clf_kwargs={**fit_kwargs}, 
        )
    else:
        history = model.fit(
            trainset, 
            epochs=epochs, 
            validation_data=valset, 
            callbacks=callbacks_list, 
            verbose=training_verbose
        ).history

    if save_weights:
        weights_path = os.path.join(
            image_callback.results_path, 
            "model.weights.h5"
        )

        if isinstance(model, dict):
            model["classifier"].save_weights(weights_path)
            replay_weights_path = os.path.join(
                image_callback.results_path, 
                "replay-model.weights.h5"
            )

            if model["generative_model"] is not None:
                model["generative_model"].save_weights(replay_weights_path)
                if config is not None:
                    config.model.weights_path = replay_weights_path
                    config.hpo["replay_weights_path"] = replay_weights_path
                    config.hpo["classifier_weights_path"] = weights_path
            elif config is not None:
                config.model.weights_path = weights_path
        else:
            model.save_weights(weights_path)
            if config is not None:
                config.model.weights_path = weights_path

    if save_config_ and config is not None:
        save_config(config, config_path)

    return history


def report(
    config: Config | None = None, 
    history=None, 
    model=None, 
    trainset=None, 
    valset=None, 
    **kwargs: object
):
    """Create configured history, evaluation, sample-image, and GIF reports.

    Args:
        config (Config | None): Reporting settings and result path.  ``None``
            displays the history and final samples without writing artifacts.
        history (Mapping[str, Sequence[float]]): Epoch metric series, normally
            the return value of :func:`train_model`.
        model (tf.keras.Model | dict): Trained model or continual mapping.
        trainset (tf.data.Dataset | Callable): Training input used for optional
            evaluation.
        valset (tf.data.Dataset | None): Validation data. Evaluation is skipped
            when it is ``None``, even if ``run_valset_eval=True``.
        **kwargs (object): Direct-mode reporting keys used only when ``config``
            is ``None``: ``results_path``, ``show_history_plot``,
            ``save_history_plot``, ``plot_without_20percent``, ``save_csv``,
            ``run_trainset_eval``, ``run_valset_eval``, ``verbose``,
            ``dataset_name``, ``show_final_images``, ``save_final_images``,
            ``save_final_gifs``, ``final_images_steps``, and
            ``final_images_cfg_scale``.

    Returns:
        dict: Evaluation metrics. Requested artifacts are displayed and/or
        written beneath the result path.

    Raises:
        TypeError: If a saving option is enabled without a usable result path.
        ValueError: If requested sampling steps/scales or datasets violate the
            wrapper's constraints.
    """

    if config is None:
        results_path = kwargs.get("results_path", "./results")
        save_history_plot = kwargs.get("save_history_plot", False)
        save_csv = kwargs.get("save_csv", False)
        show_history_plot = kwargs.get("show_history_plot", True)
        plot_without_20percent = kwargs.get("plot_without_20percent", True)
        run_trainset_eval = kwargs.get("run_trainset_eval", True)
        verbose = kwargs.get("verbose", kwargs.get("training_verbose", True))
        run_valset_eval = kwargs.get("run_valset_eval", True)
        dataset_name = kwargs.get("dataset_name", "mnist")
        save_final_images = kwargs.get("save_final_images", False)
        show_final_images = kwargs.get("show_final_images", True)
        final_images_steps = kwargs.get("final_images_steps", 1_000)
        final_images_cfg_scale = kwargs.get("final_images_cfg_scale", 3.0)
        save_final_gifs = kwargs.get("save_final_gifs", False)
    else:
        results_path = config.training.results_path
        save_history_plot = config.reporting.save_history_plot
        save_csv = config.reporting.save_csv
        show_history_plot = config.reporting.show_history_plot
        plot_without_20percent = config.reporting.plot_without_20percent
        run_trainset_eval = config.reporting.run_trainset_eval
        verbose = config.training.verbose
        run_valset_eval = config.reporting.run_valset_eval
        dataset_name = config.dataset.name
        save_final_images = config.reporting.save_final_images
        show_final_images = config.reporting.show_final_images
        final_images_steps = config.reporting.final_images_steps
        final_images_cfg_scale = config.reporting.final_images_cfg_scale
        save_final_gifs = config.reporting.save_final_gifs

    if save_history_plot:
        plot_save_path = os.path.join(
            results_path, 
            "train history.png"
        )
        plot_save_path_without_20percent = os.path.join(
            results_path, 
            "train history without first 20percent.png"
        )
    else:
        plot_save_path = None
        plot_save_path_without_20percent = None

    if save_csv:
        csv_train_save_path = os.path.join(
            results_path, 
            "train history.csv"
        )
        csv_evals_save_path = os.path.join(
            results_path, 
            "evals history.csv"
        )
    else:
        csv_train_save_path = None

    if history:
        plot_history(
            history, 
            show_plots=show_history_plot, 
            plot_path=plot_save_path, 
            csv_path=csv_train_save_path
        )
        if plot_without_20percent:
            epochs = len(next(iter(history.values())))
            plot_history(
                history, 
                range_=(int(0.2*epochs), None), 
                show_plots=show_history_plot, 
                plot_path=plot_save_path_without_20percent
            )

    if isinstance(model, dict): # Continual Learning
        eval_results = {
            "average_accuracy": float(np.nanmean(
                history["continual_accuracy"]
            )), 
            "final_accuracy": float(history["continual_accuracy"][-1])
        }

        if save_csv:
            pd.DataFrame([eval_results]).to_csv(
                csv_evals_save_path, 
                index=False
            )

        return eval_results

    if not isinstance(model, DiffusionModel):
        eval_results = {}

        if run_trainset_eval:
            eval_results["trainset_eval"] = model.evaluate(
                trainset, 
                return_dict=True, 
                verbose=verbose
            )

        if run_valset_eval and valset is not None:
            eval_results["valset_eval"] = model.evaluate(
                valset, 
                return_dict=True, 
                verbose=verbose
            )

        if eval_results and save_csv:
            pd.DataFrame(eval_results).T.to_csv(
                csv_evals_save_path, 
                index=True
            )

        if isinstance(model, VariationalAutoencoder) and (
            show_final_images or save_final_images
        ):
            class_num, image_shape, _ = get_dataset_spec(dataset_name)

            if model.conditioned:
                imgs, _ = model.generate(samples_per_class=1)
            else:
                imgs = model.generate(samples_per_class=class_num)

            imgs = np.asarray(imgs).reshape((-1, *image_shape))
            imgs_save_path = os.path.join(
                results_path, 
                "final-images.png"
            ) if save_final_images else None

            if (show_final_images or save_final_images) \
            and imgs.shape[1] == image_shape[0]:
                plot_images(
                    imgs, 
                    show_images=show_final_images, 
                    save_path=imgs_save_path
                )

        return eval_results


    def evaluate_diffusion(dataset, network_name):
        if isinstance(model, DiffusionClassifierV2):
            return model.evaluate(
                eval_both=True, 
                x=dataset, 
                network_name=network_name, 
                verbose=verbose
            )
        return model.evaluate(
            dataset, 
            network_name=network_name, 
            return_dict=True, 
            verbose=verbose
        )


    eval_results = {}

    if run_trainset_eval:
        print("Trainset evaluation:")

        if model.use_ema:
            print("EMA Network:")
            trainset_ema_eval = evaluate_diffusion(trainset, "ema")
            eval_results["trainset_ema_eval"] = trainset_ema_eval

        print("Raw Network:")
        trainset_network_eval = evaluate_diffusion(trainset, "raw")
        eval_results["trainset_network_eval"] = trainset_network_eval

    if run_valset_eval and valset is not None:
        print("Valset evaluation:")

        if model.use_ema:
            print("EMA Network:")
            valset_ema_eval = evaluate_diffusion(valset, "ema")
            eval_results["valset_ema_eval"] = valset_ema_eval

        print("Raw Network:")
        valset_network_eval = evaluate_diffusion(valset, "raw")
        eval_results["valset_network_eval"] = valset_network_eval

    if len(eval_results) > 0 and save_csv:
        eval_results_df = pd.DataFrame(eval_results).T
        eval_results_df.index.name = "dataset + network type"
        eval_results_df.to_csv(
            csv_evals_save_path, 
            index=True
        )

    if not any((show_final_images, save_final_images, save_final_gifs)):
        return eval_results

    if save_final_gifs and not model.swap_noise_image:
        imgs, frames1, frames2 = model.sample(
            scale=final_images_cfg_scale, 
            steps=final_images_steps, 
            return_x_ts=True, 
            return_x0s=True
        )
        create_gif(
            os.path.join(
                results_path, 
                f"final-gifs_steps-{final_images_steps}"
                f"_scale-{final_images_cfg_scale:.1f}.gif"
            ), 
            frames1, frames2
        )
    else:
        imgs = model.sample(
            network_name=network_name, 
            scale=final_images_cfg_scale, 
            steps=final_images_steps
        )
    
    if save_final_images:
        imgs_save_path = os.path.join(
            results_path, 
            f"final-images_steps-{final_images_steps}"
            f"_scale-{final_images_cfg_scale:.1f}.png"
        )
    else:
        imgs_save_path = None

    plot_images(
        imgs, 
        show_images=show_final_images, 
        save_path=imgs_save_path
    )

    return eval_results 


def main(
    config: Config | None = None, 
    **kwargs: object
) -> dict[str, object]:
    """Run dataset setup, model construction, fitting, and final reporting.

    Args:
        config (Config | None): Complete training configuration.  ``None``
            runs the same pipeline with information available in kwargs.
        **kwargs (object): Direct dataset, model, training, and reporting
            settings used when ``config`` is ``None``. Continual runs use
            ``task="continual"`` and ``continually_learn_kwargs``; calling
            :func:`common.learner.continually_learn` directly exposes the full
            legacy keyword set without nesting.

    Returns:
        dict[str, object]: ``model``, metric ``history``, final ``evaluations``,
        and the concrete timestamped ``results_path``.
    """

    if config is not None:
        print(
            "Initiating training process with "
            f"the following settings:\n{config}"
        )

        seed = config.training.seed
    else:
        print(
            "Initiating training process with "
            f"the following settings:\n{kwargs}"
        )

        seed = kwargs.get("seed")

    if seed is not None:
        utils.set_random_seed(seed)

    trainset, valset = get_datasets(
        config, 
        **kwargs
    )

    if config is None and "trainset_len" not in kwargs \
    and not callable(trainset):
        kwargs["trainset_len"] = len(trainset)

    model = get_model(
        config, 
        **kwargs
    )

    history = train_model(
        config, 
        model, 
        trainset, 
        valset=valset, 
        **kwargs
    )

    evaluations = report(
        config, 
        history, 
        model, 
        trainset, 
        valset=valset, 
        **kwargs
    )

    return {
        "model": model, 
        "history": history, 
        "evaluations": evaluations, 
        "results_path": config.training.results_path if config is not None
                        else kwargs.get("results_path", "./results")
    }


if __name__ == "__main__":
    config = load_config()
    main(config)
