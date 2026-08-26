"""Config-driven dataset, model, training, and reporting orchestration.

Raw networks from ``diffusion.models.transformer`` implement one neural
forward pass.  Wrappers from ``diffusion.models.wrapper`` own the diffusion
schedule, EMA copy, noising, custom optimization steps, evaluation, and
iterative sampling around the selected network.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import utils, callbacks

import pandas as pd

import numpy as np

import os

import json

from copy import deepcopy

from collections.abc import Callable, Mapping, Sequence

from common.utils import plot_images, plot_history, create_gif
from common.lr_logger_callback import LrLoggerCallback
from common.config import Config, load_config, save_config
from common.dataloader import get_datasets, get_dataset_spec
from common.model import get_model
from common.learner import _run_continual_tasks

from autoencoder.variational_autoencoder import VariationalAutoencoder

from diffusion.models.wrapper.diffusion_model import DiffusionModel
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2
from diffusion.callbacks.image_generator_callback import ImageGeneratorCallback


_PROGRESSIVE_FIT_KEYS = frozenset({
    "stage_tasks", "stages_num", "stages_verbose", "stage_epochs",
    "final_epochs", "timestep_boundaries", "timestep_clustering_type",
    "resolutions", "depths", "pacing_type", "earlystopping_type",
    "monitor", "patience", "min_delta", "stopper_mode",
})
"""Arguments owned by the progressive diffusion training API."""


def _resolve_training_options(
    config: Config | None,
    model: tf.keras.Model | dict[str, object],
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Resolve direct and configured training values into one flat mapping.

    Args:
        config (Config | None): Optional typed project configuration.
        model (tf.keras.Model | dict[str, object]): Selected training target.
        kwargs (Mapping[str, object]): Direct-mode keyword arguments.

    Returns:
        dict[str, object]: Independent mutable values consumed by
        :func:`train_model`.
    """

    # Preserve the established direct-mode defaults and legacy alias.
    if config is None:
        return {
            "show_images": kwargs.get("show_images", True),
            "save_gifs": kwargs.get("save_gifs", False),
            "results_path": kwargs.get("results_path", "./results"),
            "project_tag": kwargs.get("project_tag", ""),
            "report_every_epoch": kwargs.get("report_every_epoch", True),
            "patience": kwargs.get("patience", 0),
            "monitor": kwargs.get("monitor"),
            "monitor_mode": kwargs.get("monitor_mode", "auto"),
            "use_tensorboard": kwargs.get("tensorboard", False),
            "tensorboard_run_name": kwargs.get("tensorboard_run_name"),
            "tensorboard_path": kwargs.get("tensorboard_path"),
            "hpo": kwargs.get("hpo", {}),
            "continual_kwargs": deepcopy(kwargs.get(
                "continually_learn_kwargs",
                kwargs.get("continual_kwargs", {}),
            )),
            "dataset_name": kwargs.get("dataset_name", "mnist"),
            "loader_preprocess": kwargs.get("preprocess", "standardize"),
            "features_path": kwargs.get("features_path", ""),
            "onehot_labels": kwargs.get("onehot_labels", False),
            "validation_ratio": kwargs.get("validation_ratio", 0.),
            "use_valset": kwargs.get("use_valset", True),
            "seed": kwargs.get("seed"),
            "training_verbose": kwargs.get("verbose", 1),
            "epochs": kwargs.get("epochs", 20),
            "batch_size": kwargs.get("batch_size", 128),
            "continual_return_features": kwargs.get("return_features"),
            "continual_max_train_samples": kwargs.get("max_train_samples"),
            "continual_max_val_samples": kwargs.get("max_val_samples"),
            "continual_shuffle_buffer": kwargs.get("shuffle_buffer"),
            "continual_pad": kwargs.get("pad", 0),
            "initial_classifier": kwargs.get("initial_classifier"),
            "save_weights": kwargs.get("save_weights", False),
            "classifier_compile_overrides": deepcopy(
                kwargs.get("classifier_kwargs", {}).get("compile_args", {})
            ),
            "fit_method": kwargs.get("fit_method", "fit"),
            "fit_kwargs": dict(kwargs.get("fit_kwargs", {})),
        }

    initial_classifier = None
    # Resume the standalone continual classifier when no replay model exists.
    if isinstance(model, dict) \
    and model.get("generative_model") is None \
    and config.model.weights_path is not None:
        initial_classifier = model.get("classifier")

    return {
        "show_images": config.training.show_images,
        "save_gifs": config.training.save_gifs,
        "results_path": config.training.results_path,
        "project_tag": config.training.project_tag,
        "report_every_epoch": config.training.report_every_epoch,
        "patience": config.training.patience,
        "monitor": config.training.monitor,
        "monitor_mode": config.training.monitor_mode,
        "use_tensorboard": config.training.tensorboard,
        "tensorboard_run_name": config.training.tensorboard_run_name,
        "tensorboard_path": config.training.tensorboard_path,
        "hpo": config.hpo,
        "continual_kwargs": config.continually_learn.kwargs(),
        "dataset_name": config.dataset.name,
        "loader_preprocess": config.dataset.preprocess,
        "features_path": config.dataset.features_path,
        "onehot_labels": config.dataset.onehot_labels,
        "validation_ratio": config.dataset.validation_ratio,
        "use_valset": config.training.use_valset,
        "seed": config.training.seed,
        "training_verbose": config.training.verbose,
        "epochs": config.training.epochs,
        "batch_size": config.dataset.batch_size,
        "continual_return_features": config.dataset.return_features,
        "continual_max_train_samples": config.dataset.max_train_samples,
        "continual_max_val_samples": config.dataset.max_val_samples,
        "continual_shuffle_buffer": config.dataset.shuffle_buffer,
        "continual_pad": config.dataset.pad,
        "initial_classifier": initial_classifier,
        "save_weights": config.training.save_weights,
        "classifier_compile_overrides": deepcopy(
            config.model.classifier_kwargs.get("compile_args", {})
        ),
        "fit_method": config.training.fit_method,
        "fit_kwargs": dict(config.training.fit_kwargs),
    }


def _resolve_reporting_options(
    config: Config | None,
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Resolve direct and configured reporting values into one flat mapping.

    Args:
        config (Config | None): Optional typed project configuration.
        kwargs (Mapping[str, object]): Direct-mode keyword arguments.

    Returns:
        dict[str, object]: Independent reporting options consumed by
        :func:`report`.
    """

    # Preserve the interactive defaults used by direct notebook calls.
    if config is None:
        return {
            "results_path": kwargs.get("results_path", "./results"),
            "save_history_plot": kwargs.get("save_history_plot", False),
            "save_csv": kwargs.get("save_csv", False),
            "show_history_plot": kwargs.get("show_history_plot", True),
            "plot_without_20percent": kwargs.get(
                "plot_without_20percent", True
            ),
            "run_trainset_eval": kwargs.get("run_trainset_eval", True),
            "verbose": kwargs.get(
                "verbose", kwargs.get("training_verbose", True)
            ),
            "run_valset_eval": kwargs.get("run_valset_eval", True),
            "evaluate_ensemble_accuracy": kwargs.get(
                "evaluate_ensemble_accuracy", False
            ),
            "ensemble_accuracy_kwargs": dict(
                kwargs.get("ensemble_accuracy_kwargs") or {}
            ),
            "dataset_name": kwargs.get("dataset_name", "mnist"),
            "save_final_images": kwargs.get("save_final_images", False),
            "show_final_images": kwargs.get("show_final_images", True),
            "final_images_steps": kwargs.get("final_images_steps", 1_000),
            "final_images_cfg_scale": kwargs.get(
                "final_images_cfg_scale", 3.0
            ),
            "save_final_gifs": kwargs.get("save_final_gifs", False),
        }

    return {
        "results_path": config.training.results_path,
        "save_history_plot": config.reporting.save_history_plot,
        "save_csv": config.reporting.save_csv,
        "show_history_plot": config.reporting.show_history_plot,
        "plot_without_20percent": config.reporting.plot_without_20percent,
        "run_trainset_eval": config.reporting.run_trainset_eval,
        "verbose": config.training.verbose,
        "run_valset_eval": config.reporting.run_valset_eval,
        "evaluate_ensemble_accuracy": (
            config.reporting.evaluate_ensemble_accuracy
        ),
        "ensemble_accuracy_kwargs": dict(
            config.reporting.ensemble_accuracy_kwargs or {}
        ),
        "dataset_name": config.dataset.name,
        "save_final_images": config.reporting.save_final_images,
        "show_final_images": config.reporting.show_final_images,
        "final_images_steps": config.reporting.final_images_steps,
        "final_images_cfg_scale": config.reporting.final_images_cfg_scale,
        "save_final_gifs": config.reporting.save_final_gifs,
    }


def _evaluate_diffusion(
    model: DiffusionModel,
    dataset: tf.data.Dataset,
    network_name: str,
    verbose: int | bool,
    evaluate_ensemble_accuracy: bool,
    ensemble_accuracy_kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate one raw or EMA diffusion network on one dataset.

    Args:
        model (DiffusionModel): Diffusion wrapper selected for evaluation.
        dataset (tf.data.Dataset): Batched evaluation input.
        network_name (str): ``"raw"`` or ``"ema"`` network selector.
        verbose (int | bool): Keras evaluation verbosity.
        evaluate_ensemble_accuracy (bool): Add timestep-ensemble accuracy.
        ensemble_accuracy_kwargs (Mapping[str, object]): Ensemble options.

    Returns:
        dict[str, object]: Wrapper metrics with optional ensemble accuracy.
    """

    # Ask V2 wrappers to evaluate generator and classifier together.
    if isinstance(model, DiffusionClassifierV2):
        results = model.evaluate(
            eval_both=True,
            x=dataset,
            network_name=network_name,
            verbose=verbose,
        )
    # Use the standard wrapper evaluation path for other diffusion models.
    else:
        results = model.evaluate(
            dataset,
            network_name=network_name,
            return_dict=True,
            verbose=verbose,
        )

    # Add ensemble accuracy beside the ordinary evaluation metrics.
    if evaluate_ensemble_accuracy and isinstance(
        model, DiffusionClassifier
    ):
        selected_kwargs = dict(ensemble_accuracy_kwargs)
        selected_kwargs["netwrok_name"] = network_name
        selected_kwargs.setdefault("verbose", bool(verbose))
        results["ensemble_accuracy"] = model.evaluate_ensemble_accuracy(
            dataset, **selected_kwargs
        )

    return results


def _plain_config_value(value: object) -> object:
    """Convert tracked Keras containers into YAML-safe built-in containers.

    Args:
        value (object): One value from a model's serializable constructor
            configuration.

    Returns:
        object: Recursively copied dictionaries/lists with scalar leaves.
    """

    # Convert TensorFlow dictionary wrappers and ordinary mappings alike.
    if isinstance(value, Mapping):
        return {
            key: _plain_config_value(item)
            for key, item in value.items()
        }
    # Give unordered constructor collections a plain YAML sequence form.
    if isinstance(value, (set, frozenset)):
        return [_plain_config_value(item) for item in value]
    # Convert TensorFlow list wrappers, tuples, and ordinary sequences.
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain_config_value(item) for item in value]

    return value


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
            after paths are updated. Dynamic diffusion weight saving always
            writes the required paired config even when this is false. It does
            not disable callback artifacts or other weight saving.
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
            shallow-copied and forwarded to that method. Configured runs use
            ``training.fit_method`` and the progressive fields on
            :class:`TrainingConfig`; ``fit_progressively`` uses stage/final
            epochs instead of ``training.epochs``. A continual model
            bundle consumes ``continually_learn_kwargs`` (the legacy
            ``continual_kwargs`` spelling remains accepted in direct mode) and
            also accepts ``max_train_samples``,
            ``max_val_samples``, ``shuffle_buffer``, raw-image ``pad``, and 
            ``seed``.

    Returns:
        dict[str, list[float]]: The Keras ``History.history`` mapping with one
        value per completed epoch for each reported metric. Continual bundles
        return ``continual_accuracy`` and ``task_val_accuracy``, plus
        ``continual_ensemble_accuracy`` when enabled.

    Side Effects:
        Sets ``config.training.results_path`` from
        ``ImageGeneratorCallback.results_path``. If weight saving is enabled,
        writes ``model.weights.h5`` and sets ``config.model.weights_path`` to
        the selected model's weights. A continual replay model is saved as
        ``replay-model.weights.h5`` and is the selected model in that case.
        Dynamic diffusion saves also persist the current raw ``num_classes``
        and wrapper ``seen_classes`` in the mandatory paired ``config.yaml``.
        Progressive weight saves likewise force a final config rewrite and
        record the post-growth network constructor state.
        TensorBoard and HPO metadata are written when configured. Epoch
        trajectory sampling is limited to compatible diffusion wrappers. A
        continual model bundle is updated with its final classifier/replay
        model and a ``continual_details`` mapping.

    Raises:
        TypeError: If ``model`` or ``trainset`` is omitted, or dynamic
            diffusion weights are requested without a ``Config`` to save, or
            a direct ``fit_method`` is not a string.
        ValueError: If configured fit selection/arguments conflict, a
            progressive curriculum lacks ``stage_tasks``, or its target is not
            a diffusion wrapper.
    """

    # Require both a model and training input for orchestration.
    if model is None or trainset is None:
        raise TypeError("model and trainset are required.")

    training_options = _resolve_training_options(config, model, kwargs)
    show_images = training_options["show_images"]
    save_gifs = training_options["save_gifs"]
    results_path = training_options["results_path"]
    project_tag = training_options["project_tag"]
    report_every_epoch = training_options["report_every_epoch"]
    patience = training_options["patience"]
    monitor = training_options["monitor"]
    monitor_mode = training_options["monitor_mode"]
    use_tensorboard = training_options["use_tensorboard"]
    tensorboard_run_name = training_options["tensorboard_run_name"]
    tensorboard_path = training_options["tensorboard_path"]
    hpo = training_options["hpo"]
    continual_kwargs = training_options["continual_kwargs"]
    dataset_name = training_options["dataset_name"]
    loader_preprocess = training_options["loader_preprocess"]
    features_path = training_options["features_path"]
    onehot_labels = training_options["onehot_labels"]
    validation_ratio = training_options["validation_ratio"]
    use_valset = training_options["use_valset"]
    seed = training_options["seed"]
    training_verbose = training_options["training_verbose"]
    epochs = training_options["epochs"]
    batch_size = training_options["batch_size"]
    continual_return_features = training_options[
        "continual_return_features"
    ]
    continual_max_train_samples = training_options[
        "continual_max_train_samples"
    ]
    continual_max_val_samples = training_options[
        "continual_max_val_samples"
    ]
    continual_shuffle_buffer = training_options["continual_shuffle_buffer"]
    continual_pad = training_options["continual_pad"]
    initial_classifier = training_options["initial_classifier"]
    save_weights = training_options["save_weights"]
    classifier_compile_overrides = training_options[
        "classifier_compile_overrides"
    ]
    fit_method = training_options["fit_method"]
    fit_kwargs = training_options["fit_kwargs"]
    is_continual = isinstance(model, dict)

    # Apply stricter selector and fit-ownership checks to typed configurations.
    if config is not None:
        # Restrict the typed selector to the two documented training methods.
        if fit_method not in ("fit", "fit_progressively"):
            raise ValueError(
                "training.fit_method must be 'fit' or 'fit_progressively'."
            )

        orchestration_fit_keys = {
            "x", "y", "epochs", "initial_epoch", "validation_data",
            "callbacks", "verbose",
        }
        conflicting_fit_keys = sorted(
            orchestration_fit_keys.intersection(fit_kwargs)
        )
        # Keep data, epoch, callback, and verbosity ownership unambiguous.
        if conflicting_fit_keys:
            raise ValueError(
                "training.fit_kwargs cannot override train_model arguments: "
                + str(conflicting_fit_keys)
            )

        # Assemble the existing progressive API from its typed config fields.
        if fit_method == "fit_progressively":
            # Require an explicit curriculum instead of inventing hidden stages.
            if config.training.stage_tasks is None:
                raise ValueError(
                    "training.stage_tasks is required when "
                    "fit_method='fit_progressively'."
                )

            conflicting_fit_keys = sorted(
                _PROGRESSIVE_FIT_KEYS.intersection(fit_kwargs)
            )
            # Keep named progressive arguments in their explicit config fields.
            if conflicting_fit_keys:
                raise ValueError(
                    "training.fit_kwargs cannot override configured "
                    "progressive arguments: "
                    + str(conflicting_fit_keys)
                )

            fit_kwargs = {
                "stage_tasks": config.training.stage_tasks,
                "stages_num": config.training.stages_num,
                "stages_verbose": config.training.stages_verbose,
                "stage_epochs": config.training.stage_epochs,
                "final_epochs": config.training.final_epochs,
                "timestep_boundaries": config.training.timestep_boundaries,
                "timestep_clustering_type": (
                    config.training.timestep_clustering_type
                ),
                "resolutions": config.training.resolutions,
                "depths": config.training.depths,
                "pacing_type": config.training.pacing_type,
                "earlystopping_type": config.training.earlystopping_type,
                "monitor": config.training.progressive_monitor,
                "patience": config.training.progressive_patience,
                "min_delta": config.training.min_delta,
                "stopper_mode": config.training.stopper_mode,
                **fit_kwargs,
            }

    # Require a string before using the shared method-name convention.
    if not isinstance(fit_method, str):
        raise TypeError("fit_method must be a string.")

    progressive_fit = fit_method.endswith("progressively")

    checkpoint_model = model.get("generative_model") \
        if is_continual else model
    # Reject unsupported progressive targets before callbacks or files exist.
    if progressive_fit and not isinstance(checkpoint_model, DiffusionModel):
        raise ValueError(
            "Progressive fitting requires a DiffusionModel wrapper."
        )

    dynamic_diffusion_checkpoint = save_weights and isinstance(
        checkpoint_model, DiffusionModel
    ) and getattr(
        checkpoint_model.network, "dynamic_num_classes", False
    )
    # Never write dynamic diffusion weights without their reconstruction state.
    if dynamic_diffusion_checkpoint and config is None:
        raise TypeError(
            "Dynamic diffusion weight saving requires a Config so its "
            "num_classes and seen_classes can be saved."
        )
    # A reconstructable dynamic checkpoint always includes its paired YAML.
    if dynamic_diffusion_checkpoint:
        save_config_ = True
    # Persist the final serializable topology beside progressive weights.
    if save_weights and progressive_fit and config is not None:
        save_config_ = True

    base_callbacks = [
        LrLoggerCallback(), 
        callbacks.ProgbarLogger(count_mode="steps")
    ]
    callbacks_list = list(base_callbacks)
    forwarded_callbacks = []

    image_callback = ImageGeneratorCallback(
        show_images=show_images, 
        save_gifs=save_gifs, 
        results_path=results_path, 
        project_tag=project_tag
    )
    # Sample diffusion trajectories only for compatible wrappers.
    if isinstance(model, DiffusionModel) and \
    report_every_epoch and not model.swap_noise_image:
        callbacks_list.append(image_callback)

    # Add ordinary early stopping outside continual bundle training.
    if not is_continual and patience > 0 and not progressive_fit:
        callbacks_list.append(callbacks.EarlyStopping(
            monitor=monitor or ("val_loss" if valset is not None else "loss"),
            mode=monitor_mode, 
            patience=patience, 
            restore_best_weights=True
        ))

    # Record the callback's concrete timestamped results directory.
    if config is not None:
        config.training.results_path = image_callback.results_path

    # Configure TensorBoard metrics and hyperparameter metadata when requested.
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

        tensorboard_callback = callbacks.TensorBoard(
            log_dir=tensorboard_path, 
            histogram_freq=0, 
            write_graph=False
        )
        callbacks_list.append(tensorboard_callback)
        forwarded_callbacks.append(tensorboard_callback)

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

    # Append caller-provided callbacks after the shared defaults.
    if extra_callbacks is not None:
        extra_callbacks = list(extra_callbacks)
        callbacks_list += extra_callbacks
        forwarded_callbacks += extra_callbacks

    # Write the resolved configuration before training when requested.
    if save_config_ and config is not None:
        config_path = os.path.join(
            image_callback.results_path, 
            "config.yaml"
        )
        save_config(config, config_path)

    progressive_call_kwargs = {
        "x": trainset,
        "validation_data": valset,
        "callbacks": callbacks_list,
        "verbose": training_verbose,
        **fit_kwargs,
    }
    standard_fit_kwargs = {
        "x": trainset,
        "epochs": epochs,
        "validation_data": valset,
        "callbacks": callbacks_list,
        "verbose": training_verbose,
        **fit_kwargs,
    }

    # Delegate continual bundles to the incremental-learning workflow.
    if is_continual:
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
        # Require at least the initial two-class continual task.
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
            "callback_monitor_mode",
            "fit_method",
            "fit_kwargs"
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

        details = _run_continual_tasks(
            class_num=class_num, 
            load_dataset_fn=trainset, 
            load_dataset_fn_kwargs=loader_kwargs, 
            tuned_model_path=template_path, 
            use_loaded_opt=use_loaded_opt, 
            batch_size=batch_size, 
            epochs=epochs, 
            generative_model=None if use_buffer else model["generative_model"], 
            generative_model_kwargs=generative_kwargs, 
            callbacks_list=forwarded_callbacks,
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
            fit_method=fit_method,
            fit_kwargs=fit_kwargs,
            verbose=training_verbose, 
            **continual_kwargs
        )

        model["classifier"] = details["model"]
        model["generative_model"] = details["generative_model"]
        model["continual_details"] = details
        history = {"continual_accuracy": details["accuracies"]}
        # Preserve optional ensemble scores beside ordinary continual accuracy.
        if details["ensemble_accuracies"]:
            history["continual_ensemble_accuracy"] = details[
                "ensemble_accuracies"
            ]
        final_val_accuracy = [
            next((
                task_history[name][-1]
                for name in (
                    "val_total_accuracy",
                    "val_classifier_accuracy",
                    "val_cls_token_accuracy",
                    "val_avg_pooling_accuracy",
                    "val_accuracy",
                )
                if task_history.get(name)
            ), np.nan)
            for task_history in details["histories"]
        ]
        history["task_val_accuracy"] = final_val_accuracy
    # Train V2's generator progressively, then retain its ordinary classifier
    # phase so the configured selector mirrors the existing combined fit.
    elif fit_method == "fit_progressively" and isinstance(
        model, DiffusionClassifierV2
    ):
        discriminator_kwargs = dict(fit_kwargs)
        # Remove curriculum-only values before the ordinary discriminator fit.
        for name in _PROGRESSIVE_FIT_KEYS:
            discriminator_kwargs.pop(name, None)

        generator_history = model.fit_generator_progressively(
            **progressive_call_kwargs
        ).history
        discriminator_history = model.fit_discriminator(
            x=trainset,
            epochs=epochs,
            validation_data=valset,
            callbacks=callbacks_list,
            verbose=training_verbose,
            **discriminator_kwargs,
        ).history
        history = model._merge_result_dicts(
            (generator_history, discriminator_history),
            ("generator", "discriminator"),
        )
    # Invoke a caller-selected training method when it is not standard fit.
    elif fit_method != "fit":
        method = getattr(model, fit_method)

        # Adapt shared arguments to the project's custom train API.
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
        # Progressive methods own their stage and final epoch budgets.
        elif progressive_fit:
            trained = method(
                **progressive_call_kwargs
            )
        # Adapt shared arguments to other Keras-like training methods.
        else:
            trained = method(
                **standard_fit_kwargs
            )

        history = getattr(trained, "history", trained)
    # Train V2 generator and classifier phases with separate fit mappings.
    elif isinstance(model, DiffusionClassifierV2):
        history = model.fit(
            gen_kwargs=standard_fit_kwargs,
            clf_kwargs=dict(standard_fit_kwargs),
        )
    # Use ordinary Keras fit for remaining model families.
    else:
        history = model.fit(**standard_fit_kwargs).history

    # Persist final trained weights when requested.
    if save_weights:
        weights_path = os.path.join(
            image_callback.results_path, 
            "model.weights.h5"
        )

        checkpoint_model = model.get("generative_model") \
            if is_continual else model
        # Store the final progressive network topology before saving weights.
        if progressive_fit and config is not None and isinstance(
            checkpoint_model, DiffusionModel
        ):
            network_config = dict(_plain_config_value(
                checkpoint_model.network.get_config()
            ))
            configured_model_name = str(
                config.model.name or (
                    "dit_classifier" if config.model.with_classifier
                    else "diffusion_transformer"
                )
            ).lower()

            # Give a named model an exact constructor mapping on reload.
            if config.model.name is not None:
                config.model.kwargs = network_config
            # Update only supported fields on the compact typed model section.
            else:
                raw_config = getattr(config.model, configured_model_name)
                for name in raw_config.__dataclass_fields__:
                    # Copy constructor values represented by the typed section.
                    if name in network_config:
                        setattr(
                            raw_config,
                            name,
                            deepcopy(network_config[name]),
                        )

        # Store the topology and label map required to reload dynamic weights.
        if dynamic_diffusion_checkpoint:
            current_num_classes = checkpoint_model.network.num_classes or None
            configured_model_name = str(
                config.model.name or (
                    "dit_classifier" if config.model.with_classifier
                    else "diffusion_transformer"
                )
            ).lower()

            # Update the same raw-model source that get_model will read.
            if config.model.name is not None and config.model.kwargs:
                raw_config = config.model.kwargs
                raw_config["num_classes"] = current_num_classes
                decoder_config = raw_config.get("decoder_kwargs")
                # Keep an explicitly configured composite decoder aligned.
                if isinstance(decoder_config, dict):
                    decoder_config["num_classes"] = current_num_classes
            # Otherwise update the selected typed raw-model section.
            else:
                raw_config = getattr(config.model, configured_model_name)
                raw_config.num_classes = current_num_classes
                decoder_config = getattr(raw_config, "decoder_kwargs", None)
                # Keep an explicitly configured composite decoder aligned.
                if isinstance(decoder_config, dict):
                    decoder_config["num_classes"] = current_num_classes

            saved_seen_classes = dict(checkpoint_model.seen_classes)
            # Update a generic wrapper mapping only when it already has precedence.
            if config.model.name is not None and config.model.wrapper_kwargs:
                config.model.wrapper_kwargs["seen_classes"] = saved_seen_classes
            # Otherwise update the typed section matching the actual wrapper.
            else:
                wrapper_section_name = "diffusion_classifier_v2" \
                    if isinstance(checkpoint_model, DiffusionClassifierV2) \
                    else "diffusion_classifier" \
                    if isinstance(checkpoint_model, DiffusionClassifier) \
                    else "diffusion_model"
                getattr(
                    config.model, wrapper_section_name
                ).seen_classes = saved_seen_classes

        # Save continual classifier and optional replay-model weights separately.
        if is_continual:
            model["classifier"].save_weights(weights_path)
            replay_weights_path = os.path.join(
                image_callback.results_path, 
                "replay-model.weights.h5"
            )

            # Persist replay-model weights when generative replay was used.
            if model["generative_model"] is not None:
                model["generative_model"].save_weights(replay_weights_path)
                # Record both continual artifact paths in configuration metadata.
                if config is not None:
                    config.model.weights_path = replay_weights_path
                    config.hpo["replay_weights_path"] = replay_weights_path
                    config.hpo["classifier_weights_path"] = weights_path
            # Record classifier weights when no replay model exists.
            elif config is not None:
                config.model.weights_path = weights_path
        # Save a standalone model's weights directly.
        else:
            model.save_weights(weights_path)
            # Record the standalone model's weight path.
            if config is not None:
                config.model.weights_path = weights_path

    # Rewrite configuration after training to include resolved artifact paths.
    if save_config_ and config is not None:
        save_config(config, config_path)

    return history


def report(
    config: Config | None = None, 
    history: Mapping[str, Sequence[float]] | None = None, 
    model: tf.keras.Model | dict[str, object] | None = None, 
    trainset: tf.data.Dataset | Callable[..., object] | None = None, 
    valset: tf.data.Dataset | None = None, 
    **kwargs: object
) -> dict[str, object]:
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
            ``evaluate_ensemble_accuracy``, ``ensemble_accuracy_kwargs``,
            ``dataset_name``, ``show_final_images``, ``save_final_images``,
            ``save_final_gifs``, ``final_images_steps``, and
            ``final_images_cfg_scale``.

    Returns:
        dict: Evaluation metrics. Enabled diffusion-classifier evaluations
        contain normal metrics plus ``ensemble_accuracy`` for each raw/EMA
        network. Requested artifacts are displayed and/or written beneath the
        result path.

    Raises:
        TypeError: If history, model, or trainset is omitted, or a saving
            option is enabled without a usable result path.
        ValueError: If requested sampling steps/scales or datasets violate the
            wrapper's constraints.
    """

    # Require completed training inputs before producing a report.
    if history is None or model is None or trainset is None:
        raise TypeError("history, model, and trainset are required.")

    reporting_options = _resolve_reporting_options(config, kwargs)
    results_path = reporting_options["results_path"]
    save_history_plot = reporting_options["save_history_plot"]
    save_csv = reporting_options["save_csv"]
    show_history_plot = reporting_options["show_history_plot"]
    plot_without_20percent = reporting_options["plot_without_20percent"]
    run_trainset_eval = reporting_options["run_trainset_eval"]
    verbose = reporting_options["verbose"]
    run_valset_eval = reporting_options["run_valset_eval"]
    evaluate_ensemble_accuracy = reporting_options[
        "evaluate_ensemble_accuracy"
    ]
    ensemble_accuracy_kwargs = reporting_options[
        "ensemble_accuracy_kwargs"
    ]
    dataset_name = reporting_options["dataset_name"]
    save_final_images = reporting_options["save_final_images"]
    show_final_images = reporting_options["show_final_images"]
    final_images_steps = reporting_options["final_images_steps"]
    final_images_cfg_scale = reporting_options["final_images_cfg_scale"]
    save_final_gifs = reporting_options["save_final_gifs"]
    is_continual = isinstance(model, dict)

    # Restrict ensemble evaluation to wrappers that expose the classifier API.
    if evaluate_ensemble_accuracy and not isinstance(
        model, (dict, DiffusionClassifier)
    ):
        raise ValueError(
            "evaluate_ensemble_accuracy requires "
            "DiffusionClassifier or DiffusionClassifierV2."
        )

    # Prepare history-plot output paths when saving is enabled.
    if save_history_plot:
        plot_save_path = os.path.join(
            results_path, 
            "train history.png"
        )
        plot_save_path_without_20percent = os.path.join(
            results_path, 
            "train history without first 20percent.png"
        )
    # Disable history-plot file output when saving is off.
    else:
        plot_save_path = None
        plot_save_path_without_20percent = None

    # Prepare training and evaluation CSV paths when saving is enabled.
    if save_csv:
        csv_train_save_path = os.path.join(
            results_path, 
            "train history.csv"
        )
        csv_evals_save_path = os.path.join(
            results_path, 
            "evals history.csv"
        )
    # Disable training-history CSV output when saving is off.
    else:
        csv_train_save_path = None

    # Plot nonempty training history using the selected outputs.
    if history:
        plot_history(
            history, 
            show_plots=show_history_plot, 
            plot_path=plot_save_path, 
            csv_path=csv_train_save_path
        )
        # Optionally plot a second view excluding the first training fifth.
        if plot_without_20percent:
            epochs = len(next(iter(history.values())))
            plot_history(
                history, 
                range_=(int(0.2*epochs), None), 
                show_plots=show_history_plot, 
                plot_path=plot_save_path_without_20percent
            )

    # Summarize continual bundles from their recorded task accuracies.
    if is_continual:
        eval_results = {
            "average_accuracy": float(np.nanmean(
                history["continual_accuracy"]
            )), 
            "final_accuracy": float(history["continual_accuracy"][-1])
        }

        # Add ensemble summaries when the continual learner produced them.
        if "continual_ensemble_accuracy" in history \
        and len(history["continual_ensemble_accuracy"]) > 0:
            eval_results.update({
                "average_ensemble_accuracy": float(np.nanmean(
                    history["continual_ensemble_accuracy"]
                )), 
                "final_ensemble_accuracy": float(
                    history["continual_ensemble_accuracy"][-1]
                )
            })

        # Persist continual summary metrics when requested.
        if save_csv:
            pd.DataFrame([eval_results]).to_csv(
                csv_evals_save_path, 
                index=False
            )

        return eval_results

    # Evaluate non-diffusion models through their standard Keras API.
    if not isinstance(model, DiffusionModel):
        eval_results = {}

        # Evaluate training data when requested.
        if run_trainset_eval:
            eval_results["trainset_eval"] = model.evaluate(
                trainset, 
                return_dict=True, 
                verbose=verbose
            )

        # Evaluate available validation data when requested.
        if run_valset_eval and valset is not None:
            eval_results["valset_eval"] = model.evaluate(
                valset, 
                return_dict=True, 
                verbose=verbose
            )

        # Persist any standard-model evaluations when requested.
        if eval_results and save_csv:
            pd.DataFrame(eval_results).T.to_csv(
                csv_evals_save_path, 
                index=True
            )

        # Generate display samples for variational autoencoders when requested.
        if isinstance(model, VariationalAutoencoder) and (
            show_final_images or save_final_images
        ):
            class_num, image_shape, _ = get_dataset_spec(dataset_name)

            # Generate one example for each known conditional class.
            if model.conditioned:
                classes = model.seen_classes or list(range(model.class_num or 0))
                imgs, _ = model.generate(
                    classes=classes,
                    samples_per_class=1
                )
            # Generate one unconditional example per dataset class.
            else:
                imgs = model.generate(samples_per_class=class_num)

            imgs = np.asarray(imgs)
            expected_width = int(np.prod(image_shape))
            # Reshape valid flattened image samples before plotting.
            if imgs.ndim == 2 and len(imgs) > 0 \
            and imgs.shape[-1] == expected_width:
                imgs = imgs.reshape((-1, *image_shape))
                imgs_save_path = os.path.join(
                    results_path,
                    "final-images.png"
                ) if save_final_images else None
                plot_images(
                    imgs, 
                    show_images=show_final_images, 
                    save_path=imgs_save_path
                )

        return eval_results

    evaluation_datasets = []
    # Include training data when its evaluation is enabled.
    if run_trainset_eval:
        evaluation_datasets.append(("Trainset", "trainset", trainset))
    # Include available validation data when its evaluation is enabled.
    if run_valset_eval and valset is not None:
        evaluation_datasets.append(("Valset", "valset", valset))

    network_evaluations = []
    # Evaluate EMA first to preserve the established report order.
    if model.use_ema:
        network_evaluations.append(("EMA", "ema", "ema_eval"))
    network_evaluations.append(("Raw", "raw", "network_eval"))

    eval_results = {}
    # Evaluate each requested dataset using each available network copy.
    for dataset_title, dataset_key, dataset in evaluation_datasets:
        print(f"{dataset_title} evaluation:")
        # Keep network labels and result-key suffixes aligned.
        for network_title, network_name, result_suffix in network_evaluations:
            print(f"{network_title} Network:")
            eval_results[f"{dataset_key}_{result_suffix}"] = (
                _evaluate_diffusion(
                    model,
                    dataset,
                    network_name,
                    verbose,
                    evaluate_ensemble_accuracy,
                    ensemble_accuracy_kwargs,
                )
            )

    # Persist nonempty diffusion evaluations when requested.
    if len(eval_results) > 0 and save_csv:
        eval_results_df = pd.DataFrame(eval_results).T
        eval_results_df.index.name = "dataset + network type"
        eval_results_df.to_csv(
            csv_evals_save_path, 
            index=True
        )

    # Skip diffusion sampling when no visual artifact was requested.
    if not any((show_final_images, save_final_images, save_final_gifs)):
        return eval_results

    # Reject GIF trajectories unavailable in swapped-noise sampling mode.
    if save_final_gifs and model.swap_noise_image:
        raise ValueError(
            "Final GIF trajectories are unavailable when swap_noise_image=True."
        )

    # Sample and save full denoising trajectories for GIF output.
    if save_final_gifs and not model.swap_noise_image:
        imgs, frames1, frames2 = model.sample(
            network_name=model.test_network_name,
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
    # Sample only final images when GIF trajectories are unnecessary.
    else:
        imgs = model.sample(
            network_name=model.test_network_name,
            scale=final_images_cfg_scale, 
            steps=final_images_steps
        )
    
    # Prepare a final-image path when file output is enabled.
    if save_final_images:
        imgs_save_path = os.path.join(
            results_path, 
            f"final-images_steps-{final_images_steps}"
            f"_scale-{final_images_cfg_scale:.1f}.png"
        )
    # Disable final-image file output while retaining optional display.
    else:
        imgs_save_path = None

    # Render final samples when display or file output was requested.
    if show_final_images or save_final_images:
        plot_images(
            imgs,
            show_images=show_final_images,
            save_path=imgs_save_path
        )

    return eval_results 


def main(
    config: Config | None = None, 
    teacher_network: tf.keras.Model | None = None,
    **kwargs: object
) -> dict[str, object]:
    """Run dataset setup, model construction, fitting, and final reporting.

    Args:
        config (Config | None): Complete training configuration.  ``None``
            runs the same pipeline with information available in kwargs.
        teacher_network (tf.keras.Model | None): Runtime-only teacher forwarded
            to diffusion-classifier construction. It is not stored in config
            YAML.
        **kwargs (object): Direct dataset, model, training, and reporting
            settings used when ``config`` is ``None``. Continual runs use
            ``task="continual"`` and ``continually_learn_kwargs``; calling
            :func:`common.learner.continually_learn` directly exposes the full
            legacy keyword set without nesting.

    Returns:
        dict[str, object]: ``model``, metric ``history``, final ``evaluations``,
        and the concrete timestamped ``results_path``.
    """

    # Announce typed configuration and obtain its seed.
    if config is not None:
        print(
            "Initiating training process with "
            f"the following settings:\n{config}"
        )

        seed = config.training.seed
    # Announce direct settings and obtain their seed.
    else:
        print(
            "Initiating training process with "
            f"the following settings:\n{kwargs}"
        )

        seed = kwargs.get("seed")

    # Seed supported random generators for reproducible orchestration.
    if seed is not None:
        utils.set_random_seed(seed)

    trainset, valset = get_datasets(
        config, 
        **kwargs
    )

    # Supply direct model construction with the prepared batch count.
    if config is None and "trainset_len" not in kwargs \
    and not callable(trainset):
        kwargs["trainset_len"] = len(trainset)

    model = get_model(
        config, 
        teacher_network=teacher_network,
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


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    config = load_config()
    main(config)
