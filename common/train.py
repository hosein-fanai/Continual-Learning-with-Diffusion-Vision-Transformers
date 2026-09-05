"""Orchestrate Config/direct-mode dataset loading, training, and reporting.

``main`` installs the runtime seed and dtype policy before loading datasets and
constructing models, then delegates to ``train_model`` and ``report``. Raw
diffusion networks implement a neural forward pass; their wrappers own noising,
schedules, EMA, optimization, evaluation, and iterative generation. This module
chooses the appropriate wrapper training/reporting entry point.

Ordinary runs pass a compiled Keras model plus datasets. Continual runs pass a
classifier/generator bundle plus a callable loader; task scheduling, replay,
distillation, and per-task evaluation are delegated to ``common.learner``.
Config calls may update the supplied configuration with materialized schedules,
grown network topology, concrete artifact directories, and weight paths.
Direct calls retain historical keyword defaults and return their concrete
artifact directory through ``main``. File output, callbacks, plotting, and
sampling follow their explicit training/reporting switches."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import callbacks

import numpy as np

import pandas as pd

import os

import json

from copy import deepcopy

from collections.abc import Callable, Mapping, Sequence

from common.utils import plot_images, plot_history, create_gif
from common.lr_logger_callback import LrLoggerCallback
from common.config import (
    Config,
    normalize_training_task,
    load_config,
    resolve_continual_schedule,
    save_config
)
from common.dataloader import get_datasets, get_dataset_spec
from common.model import get_model
from common.runtime import configure_runtime, derive_seed, effective_seed
from common.recovery import find_latest_task_checkpoint, load_task_checkpoint
from common.continual_reporting import (
    observed_mean,
    write_continual_csv_artifacts,
    write_continual_tensorboard_summaries
)

from autoencoder.variational_autoencoder import VariationalAutoencoder

from diffusion.models.wrapper.diffusion_model import DiffusionModel
from diffusion.models.wrapper.diffusion_classifier import DiffusionClassifier
from diffusion.models.wrapper.diffusion_classifier_v2 import DiffusionClassifierV2
from diffusion.callbacks.image_generator_callback import ImageGeneratorCallback


_PROGRESSIVE_FIT_KEYS = frozenset({
    "stage_tasks", "stages_num", "stages_verbose", "stage_epochs",
    "final_epochs", "timestep_boundaries", "timestep_clustering_type",
    "resolutions", "depths", "pacing_type", "earlystopping_type",
    "monitor", "patience", "min_delta", "stopper_mode"
})
"""Arguments owned by the progressive diffusion training API."""


def _normalize_results_path(
    results_path: object,
    required_for: Sequence[str] = ()
) -> str | bytes | None:
    """Resolve a filesystem path and reject its absence when a consumer needs it.

    Args:
        results_path (object): String, bytes, or ``os.PathLike`` artifact root.
            None is accepted only when ``required_for`` is empty. This function
            does not create the path, resolve it absolutely, or decode byte paths.
        required_for (Sequence[str]): Enabled features requiring an artifact root.
            Default ``()`` permits None. Names are converted to text and included
            together in the missing-path error.

    Returns:
        str | bytes | None: ``os.fspath(results_path)`` for a supplied path, or
        None when no consumer requires a destination.

    Raises:
        TypeError: If a required path is None or the supplied object does not
            implement the filesystem-path protocol."""

    requirements = tuple(str(name) for name in required_for)
    # Permit no artifact root only when every active consumer can work without it.
    if results_path is None:
        # Explain all active consumers instead of failing later in os.path.join.
        if requirements:
            raise TypeError(
                "results_path is required for " + ", ".join(requirements) + "."
            )

        return None

    return os.fspath(results_path)


def _resolve_training_options(
    config: Config | None,
    model: tf.keras.Model | dict[str, object],
    kwargs: Mapping[str, object]
) -> dict[str, object]:
    """Combine typed or direct settings into the training orchestrator's options.

    Config mode reads the training, dataset, model, continual, and HPO sections;
    direct-mode keywords do not override those values. A restored standalone
    classifier is recognized from the bundle and saved weight paths so subsequent
    task heads can inherit it. Resolving options itself does not mutate the model
    or configuration.

    Args:
        config (Config | None): Typed settings, or None to use the direct keyword
            contract. Config defaults are those of its dataclass sections, which
            intentionally differ from some historical direct defaults.
        model (tf.keras.Model | dict[str, object]): Compiled model or continual
            bundle. Only bundles are inspected for ``classifier`` and
            ``generative_model`` when recovering a standalone classifier.
        kwargs (Mapping[str, object]): Direct options when config is None. Image
            switches ``show_images``/``report_every_epoch`` default True and
            ``save_gifs``/``save_weights`` default False. ``results_path`` defaults
            to ``"./results"`` and ``project_tag`` to ``""``. Early stopping uses
            ``patience=0``, ``monitor=None``, ``monitor_mode="auto"``. Logging uses
            ``tensorboard=False``, with optional ``tensorboard_run_name`` and
            ``tensorboard_path`` both defaulting to None. ``hpo`` and
            ``continually_learn_kwargs`` default to empty mappings.
            Dataset defaults are ``dataset_name="mnist"``,
            ``preprocess="standardize"``, ``features_path=""``,
            ``onehot_labels=False``, ``validation_ratio=0.0``, and ``use_valset=True``.
            Runtime defaults are ``seed=None``, the current global ``dtype_policy``,
            and ``deterministic_ops=False``. Fit defaults are ``verbose=1``,
            ``epochs=20``, ``batch_size=128``, ``fit_method="fit"``, and empty
            ``fit_kwargs``. ``return_features``, ``max_train_samples``,
            ``max_val_samples``, ``shuffle_buffer``, and ``initial_classifier``
            default None; ``pad`` defaults 0. Optional
            ``classifier_kwargs["compile_args"]`` supplies classifier compile
            overrides, otherwise an empty mapping is used.

    Returns:
        dict[str, object]: Flat resolved options consumed by ``train_model``.
        Names such as ``tensorboard``, ``preprocess``, ``verbose``, and
        ``continually_learn_kwargs`` become ``use_tensorboard``,
        ``loader_preprocess``, ``training_verbose``, and ``continual_kwargs``.
        Continual dataset controls receive ``continual_`` prefixes. The continual
        and classifier compile mappings are detached; model/callback references
        and other returned values may still refer to caller-owned objects."""

    # Preserve the established direct-mode defaults.
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
            "continual_kwargs": deepcopy(
                kwargs.get("continually_learn_kwargs", {})
            ),
            "dataset_name": kwargs.get("dataset_name", "mnist"),
            "loader_preprocess": kwargs.get("preprocess", "standardize"),
            "features_path": kwargs.get("features_path", ""),
            "onehot_labels": kwargs.get("onehot_labels", False),
            "validation_ratio": kwargs.get("validation_ratio", 0.),
            "use_valset": kwargs.get("use_valset", True),
            "seed": kwargs.get("seed"),
            "dtype_policy": kwargs.get(
                "dtype_policy",
                tf.keras.mixed_precision.global_policy().name
            ),
            "deterministic_ops": kwargs.get("deterministic_ops", False),
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
            "fit_kwargs": dict(kwargs.get("fit_kwargs", {}))
        }

    initial_classifier = None
    paired_classifier_path = config.hpo.get("classifier_weights_path")
    uses_replay_classifier = config.continually_learn.use_generative_model_classifier
    # Only continual model bundles can resume a standalone classifier from saved weights.
    resume_classifier = (
        model.get("generative_model") is None
        and config.model.weights_path is not None
        or paired_classifier_path is not None
        and not uses_replay_classifier
    ) if isinstance(model, dict) else False

    # Seed task heads from any restored standalone continual classifier.
    if resume_classifier:
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
        "seed": effective_seed(config),
        "dtype_policy": config.training.dtype_policy,
        "deterministic_ops": config.training.deterministic_ops,
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
        "fit_kwargs": dict(config.training.fit_kwargs)
    }


def _resolve_reporting_options(
    config: Config | None,
    kwargs: Mapping[str, object]
) -> dict[str, object]:
    """Resolve report controls from Config or the historical direct keyword API.

    Args:
        config (Config | None): Typed reporting/dataset/training settings. None
            selects direct mode; a supplied Config takes precedence over kwargs.
        kwargs (Mapping[str, object]): Direct report options. Defaults are
            ``results_path="./results"``, ``save_history_plot=False``,
            ``save_csv=False``, ``show_history_plot=True``,
            ``plot_without_20percent=True``, ``run_trainset_eval=True``,
            ``run_valset_eval=True``, ``evaluate_ensemble_accuracy=False``, and
            empty ``ensemble_accuracy_kwargs``. ``verbose`` defaults to
            ``training_verbose`` when supplied, otherwise True. Final sample
            defaults are ``dataset_name="mnist"``, ``save_final_images=False``,
            ``show_final_images=True``, ``final_images_steps=1000``,
            ``final_images_cfg_scale=3.0``, and ``save_final_gifs=False``.
            ``seed``/``task`` are optional inputs to ``effective_seed``; omitted
            seed behavior therefore follows the selected runtime task contract.

    Returns:
        dict[str, object]: Flat report options including the effective seed and a
        copied ensemble-options mapping. No directories, plots, evaluations, or
        model mutations are performed by this resolver."""

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
            "seed": effective_seed(
                seed=kwargs.get("seed"),
                task=kwargs.get("task")
            )
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
        "seed": effective_seed(config)
    }


def _evaluate_diffusion(
    model: DiffusionModel,
    dataset: tf.data.Dataset,
    network_name: str,
    verbose: int | bool,
    evaluate_ensemble_accuracy: bool,
    ensemble_accuracy_kwargs: Mapping[str, object]
) -> dict[str, object]:
    """Evaluate one diffusion network branch and optionally its timestep ensemble.

    Args:
        model (DiffusionModel): Trained diffusion wrapper. V2 uses
            ``evaluate(eval_both=True)`` so generator and classifier phases are
            represented; other wrappers use ordinary dictionary-valued evaluation.
        dataset (tf.data.Dataset): Batched evaluation inputs in the wrapper's
            expected format. Classification ensembles additionally require labels.
        network_name (str): Explicit ``"raw"`` or ``"ema"`` branch forwarded to
            wrapper evaluation and overriding any branch in ensemble options.
        verbose (int | bool): Keras evaluation verbosity. Its truth value is also
            the ensemble verbosity unless explicitly provided in its option mapping.
        evaluate_ensemble_accuracy (bool): Whether to add a scalar ensemble result.
            It is acted on only for DiffusionClassifier-compatible wrappers.
        ensemble_accuracy_kwargs (Mapping[str, object]): Additional ensemble
            options, such as timestep count/chunking, weighting, head coefficients,
            and seed. Copied before the branch and default verbosity are inserted.

    Returns:
        dict[str, object]: Wrapper evaluation metrics, with ``ensemble_accuracy``
        added when requested and supported. V2 may prefix colliding phase metrics.
        Evaluation resets/updates model metric state and may select its test phase;
        the supplied options mapping is unchanged.

    Raises:
        ValueError: If the wrapper rejects a branch, dataset format, or ensemble
            setting. Errors from its evaluation methods propagate."""

    # Ask V2 wrappers to evaluate generator and classifier together.
    if isinstance(model, DiffusionClassifierV2):
        results = model.evaluate(
            eval_both=True,
            x=dataset,
            network_name=network_name,
            verbose=verbose
        )
    # Use the standard wrapper evaluation path for other diffusion models.
    else:
        results = model.evaluate(
            dataset,
            network_name=network_name,
            return_dict=True,
            verbose=verbose
        )

    # Add ensemble accuracy beside the ordinary evaluation metrics.
    if evaluate_ensemble_accuracy and isinstance(
        model, DiffusionClassifier
    ):
        selected_kwargs = dict(ensemble_accuracy_kwargs)
        # Report each evaluation against its selected network.
        selected_kwargs["network_name"] = network_name
        selected_kwargs.setdefault("verbose", bool(verbose))
        results["ensemble_accuracy"] = model.evaluate_ensemble_accuracy(
            dataset, **selected_kwargs
        )

    return results


def _plain_config_value(value: object) -> object:
    """Convert tracked configuration containers to ordinary serializable containers.

    Args:
        value (object): A Keras configuration value. Mappings are copied recursively;
            sets/frozensets and non-text Sequences become lists. Strings, bytes,
            bytearrays, and other leaf values are left unchanged.

    Returns:
        object: Converted container or unchanged leaf. This removes tracked Keras
        container wrappers but does not serialize arbitrary tensors/live models or
        guarantee that unsupported leaf objects are YAML-safe. Set iteration order
        is retained as encountered rather than canonically sorted."""

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
    _run_state: dict[str, object] | None = None,
    **kwargs: object
) -> dict[str, list[float]]:
    """Train a supplied model or continual bundle and persist the requested artifacts.

    Ordinary Keras models use ``fit``. V2 uses generator/discriminator phases;
    progressive V2 training applies its curriculum to the generator and retains a
    separate ordinary classifier phase. Direct-mode custom methods are also
    routed, including array adaptation for VAE ``train``. Continual bundles save a
    classifier template, materialize a task schedule, and delegate replay/KD and
    task training to the learner before updating the bundle with final models and
    ``continual_details``.

    Args:
        config (Config | None): Typed settings. Default None selects direct mode.
            A supplied Config is updated with its concrete artifact directory,
            realized schedule, and, when weights are saved, reconstruction topology
            and weight paths. Typed fit methods are ``fit``/``fit_progressively``.
        model (tf.keras.Model | dict[str, object] | None): Compiled training target.
            Default None is a signature placeholder; this function does not build
            the model. Continual bundles provide ``classifier_name``, ``classifier``,
            and their optional ``generative_model`` as produced by ``get_model``.
        trainset (tf.data.Dataset | Callable[..., object] | None): Batched training
            data, or the continual loader callable for a bundle. Default None;
            usable training input must be supplied for the chosen model method.
        valset (tf.data.Dataset | None): Optional batched validation data. Default
            None disables ordinary validation; continual splits come from the loader.
        save_config_ (bool): Save input/resolved YAML for Config runs. Default True.
            Saving dynamic or progressive diffusion weights can force this True so
            their grown topology remains reconstructable. Direct mode has no YAML.
        extra_callbacks (Sequence[tf.keras.callbacks.Callback] | None): Extra
            callbacks appended to standard logging callbacks and forwarded to task
            phases. Default None adds none. Callback objects are reused, not cloned.
        _run_state (dict[str, object] | None): Internal mutable output holder.
            Default None. When supplied, receives the concrete ``results_path``
            so direct ``main`` calls can report into the same run directory.
        **kwargs (object): Direct-mode settings described by
            ``_resolve_training_options``, including epochs/batching, runtime seed,
            callbacks/logging, artifact switches, ``fit_method``/``fit_kwargs``, and
            ``continually_learn_kwargs``. Config mode reads its typed fields instead.
            Direct ``fit_method="train"`` adapts VAE datasets into arrays. Reserved
            orchestration arguments cannot be duplicated in typed ``fit_kwargs``.

    Returns:
        dict[str, list[float]]: Epoch histories for ordinary training, merged phase
        histories for V2, or task trajectories for continual training:
        ``continual_accuracy``, optional ``continual_ensemble_accuracy``, and
        ``task_val_accuracy``. Continual validation values use the selected matrix
        row means; when no matrix exists, available validation history keys provide
        the fallback. Missing observations remain NaN. The model is trained in place.

    Raises:
        TypeError: If an enabled file consumer has no result path, or dynamic
            diffusion weight saving lacks the Config needed to reconstruct it.
        ValueError: If typed fit controls conflict, progressive training lacks
            stages/a diffusion target, a resume schedule differs, or VAE array
            adaptation receives an empty dataset.
        FileExistsError: If a new immutable input-config artifact would overwrite
            an existing file.
        OSError: If template, configuration, logging, or weight artifacts cannot
            be written. Model/dataset/learner errors otherwise propagate."""

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
    dtype_policy = training_options["dtype_policy"]
    deterministic_ops = training_options["deterministic_ops"]
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

        reserved_fit_keys = {
            "x", "y", "epochs", "initial_epoch",
            "validation_data", "callbacks", "verbose"
        }
        # Assemble the existing progressive API from its typed config fields.
        if fit_method == "fit_progressively":
            # Require an explicit curriculum instead of inventing hidden stages.
            if config.training.stage_tasks is None:
                raise ValueError(
                    "training.stage_tasks is required when "
                    "fit_method='fit_progressively'."
                )

            reserved_fit_keys.update(_PROGRESSIVE_FIT_KEYS)

        conflicting_fit_keys = sorted(reserved_fit_keys.intersection(fit_kwargs))
        # Keep orchestration-owned arguments out of the free-form fit mapping.
        if conflicting_fit_keys:
            raise ValueError(
                "training.fit_kwargs contains reserved keys: "
                + str(conflicting_fit_keys)
            )

        # Materialize progressive controls only for the progressive method.
        if fit_method == "fit_progressively":
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

    progressive_fit = fit_method.endswith("progressively")

    # Continual bundles checkpoint their generator; ordinary runs checkpoint the supplied
    # model.
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

    result_path_consumers = []
    # The callback needs a save destination when images are not displayed.
    if not show_images:
        result_path_consumers.append("training without interactive image display")
    # GIFs are always file artifacts.
    if save_gifs:
        result_path_consumers.append("training GIF saving")
    # Final weights are written beneath the callback's concrete result path.
    if save_weights:
        result_path_consumers.append("weight saving")
    # Typed config snapshots are written beneath the result path.
    if save_config_ and config is not None:
        result_path_consumers.append("configuration saving")
    # Continual bundles always persist their classifier template.
    if is_continual:
        result_path_consumers.append("continual training")
    # An explicit TensorBoard root is independent of the result directory.
    if use_tensorboard and tensorboard_path is None:
        result_path_consumers.append("default TensorBoard logging")
    results_path = _normalize_results_path(results_path, result_path_consumers)

    base_callbacks = [
        LrLoggerCallback(),
        callbacks.ProgbarLogger(count_mode="steps")
    ]
    callbacks_list = list(base_callbacks)
    forwarded_callbacks = []
    generative_forwarded_callbacks = []

    image_callback = ImageGeneratorCallback(
        show_images=show_images,
        save_gifs=save_gifs,
        results_path=results_path,
        project_tag=project_tag,
        seed=seed
    )
    # Sample diffusion trajectories only for compatible wrappers.
    if isinstance(checkpoint_model, DiffusionModel) and \
    report_every_epoch and not checkpoint_model.swap_noise_image:
        # Continual bundles use this callback only during their compatible
        # replay-model phase, never during standalone classifier fitting.
        if is_continual:
            generative_forwarded_callbacks.append(image_callback)
        # Ordinary diffusion training uses the single shared callback list.
        else:
            callbacks_list.append(image_callback)

    # Add ordinary early stopping outside continual bundle training.
    if not is_continual and patience > 0 and not progressive_fit:
        # Default to validation loss when validation exists and training loss otherwise.
        callbacks_list.append(callbacks.EarlyStopping(
            monitor=monitor or ("val_loss" if valset is not None else "loss"),
            mode=monitor_mode,
            patience=patience,
            restore_best_weights=True
        ))

    # Publish the callback's concrete timestamped results directory.
    if _run_state is not None:
        _run_state["results_path"] = image_callback.results_path
    # Configured callers retain the established in-place path update.
    if config is not None:
        config.training.results_path = image_callback.results_path

    # Give configured continual runs a stable task-checkpoint root before the
    # initial resolved config is written. A resumed run continues the same
    # immutable checkpoint sequence unless an explicit new root was supplied.
    if is_continual and continual_kwargs.get("save_task_checkpoints", False) \
    and continual_kwargs.get("checkpoint_dir") is None:
        resume_path = continual_kwargs.get("resume_from")
        # Continue the checkpoint sequence selected by the caller.
        if resume_path is not None:
            resolved_checkpoint_dir = str(
                find_latest_task_checkpoint(resume_path).parent
            )
        # Start a new checkpoint sequence inside the result directory.
        else:
            resolved_checkpoint_dir = os.path.join(
                image_callback.results_path,
                "checkpoints",
            )
        continual_kwargs["checkpoint_dir"] = resolved_checkpoint_dir
        # Persist the resolved recovery root in typed configurations.
        if config is not None:
            config.continually_learn.checkpoint_dir = (
                resolved_checkpoint_dir
            )

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
        try:
            with writer.as_default():
                tf.summary.text(
                    "hyperparameters",
                    json.dumps(hpo.get("params", {}), sort_keys=True),
                    step=0
                )
            writer.flush()
        finally:
            # Release the event writer before task-level writers are created.
            writer.close()

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
        declared_input_path = config.hpo.get("input_config_path")

        # Ordinary runs retain one pre-training config that final artifact-path
        # rewrites cannot invalidate as a task-checkpoint recovery input. HPO
        # already supplies its own immutable per-trial input config.
        if declared_input_path is None:
            input_config_path = os.path.join(
                image_callback.results_path,
                "input_config.yaml",
            )
            config.hpo["input_config_path"] = input_config_path
            # A timestamped result directory should be fresh; refuse to replace
            # a purported immutable recovery specification if it is not.
            if os.path.exists(input_config_path):
                raise FileExistsError(
                    f"Immutable input config already exists: {input_config_path}"
                )

            save_config(config, input_config_path)

        save_config(config, config_path)

    fit_call_kwargs = {
        "x": trainset,
        "validation_data": valset,
        "callbacks": callbacks_list,
        "verbose": training_verbose,
        **fit_kwargs
    }
    standard_fit_kwargs = {
        "epochs": epochs,
        **fit_call_kwargs
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
        task_size = continual_kwargs.pop("task_size", 1)
        class_order_mode = continual_kwargs.pop("class_order_mode", "fixed")
        task_order_mode = continual_kwargs.pop("task_order_mode", "fixed")
        schedule_seed = continual_kwargs.get("seed", seed)
        resume_path = continual_kwargs.get("resume_from")

        # Reuse the authoritative materialized schedule on configured resume.
        if resume_path is not None:
            recovered_schedule = load_task_checkpoint(resume_path)
            class_order = list(recovered_schedule.class_order)
            task_groups = [
                list(group) for group in recovered_schedule.task_groups
            ]
            # Keep a configured class count compatible with the saved schedule.
            if configured_class_num is not None \
            and int(configured_class_num) != len(class_order):
                raise ValueError(
                    "Configured class_num differs from the checkpoint schedule."
                )

            requested_order = continual_kwargs.get("class_order")

            # Compare explicit order only when stochastic modes cannot alter it.
            if requested_order is not None and class_order_mode == "fixed" and \
            task_order_mode == "fixed" and list(requested_order) != class_order:
                raise ValueError(
                    "Configured class_order differs from the checkpoint schedule."
                )

            requested_groups = continual_kwargs.get("task_groups")

            # Compare explicit groups when whole-task order remains fixed.
            if requested_groups is not None and task_order_mode == "fixed" \
            and [list(group) for group in requested_groups] != task_groups:
                raise ValueError(
                    "Configured task_groups differ from the checkpoint schedule."
                )

            # Prevent a foreign checkpoint from exceeding dataset vocabulary.
            if len(class_order) > dataset_class_num:
                raise ValueError(
                    "Checkpoint schedule exceeds the selected dataset classes."
                )
        # Resolve stochastic/fixed scheduling normally for a new experiment.
        else:
            class_order, task_groups = resolve_continual_schedule(
                configured_class_num,
                continual_kwargs.get("class_order"),
                continual_kwargs.get("task_groups"),
                available_class_num=dataset_class_num,
                task_size=task_size,
                class_order_mode=class_order_mode,
                task_order_mode=task_order_mode,
                seed=schedule_seed
            )

        class_num = len(class_order)
        # Forward one canonical schedule to the lower-level continual loop.
        continual_kwargs["class_order"] = class_order
        continual_kwargs["task_groups"] = task_groups
        # The schedule is already materialized above. Prevent the lower-level
        # API from randomizing it again or rejecting explicit groups under a
        # stale stochastic mode; the original request remains in metadata.
        continual_kwargs["class_order_mode"] = "fixed"
        continual_kwargs["task_order_mode"] = "fixed"
        # Materialize stochastic schedules in the final typed configuration.
        if config is not None:
            config.hpo.setdefault("schedule_request", {
                "task_size": task_size,
                "class_order_mode": class_order_mode,
                "task_order_mode": task_order_mode,
                "seed": schedule_seed
            })
            # The final saved config is independently reproducible: it carries
            # the realized stochastic schedule rather than relying on a future
            # Python shuffle implementation.
            config.continually_learn.class_num = class_num
            config.continually_learn.class_order = list(class_order)
            config.continually_learn.task_groups = [
                list(group) for group in task_groups
            ]
            config.continually_learn.class_order_mode = "fixed"
            config.continually_learn.task_order_mode = "fixed"

        generative_kwargs = continual_kwargs.pop(
            "generative_model_kwargs", {}
        )
        use_buffer = continual_kwargs.get("use_buffer", False)
        baseline_name = str(continual_kwargs.get("baseline") or "").lower()
        no_generator_baselines = {"sequential", "cumulative", "reservoir_er"}
        omit_generative_model = use_buffer or (
            baseline_name in no_generator_baselines
        )
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
            "initial_classifier",
            "callback_patience",
            "callback_monitor",
            "callback_monitor_mode",
            "fit_method",
            "fit_kwargs",
            "seed",
        ):
            continual_kwargs.pop(factory_owned_key, None)

        # Choose categorical loss for one-hot targets and sparse categorical loss for integer
        # IDs.
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

        from common.learner import _run_continual_tasks

        # Omit the generator for buffer/classifier-only controls; pass the bundle's generator
        # otherwise.
        details = _run_continual_tasks(
            class_num=class_num,
            load_dataset_fn=trainset,
            load_dataset_fn_kwargs=loader_kwargs,
            tuned_model_path=template_path,
            use_loaded_opt=use_loaded_opt,
            batch_size=batch_size,
            epochs=epochs,
            generative_model=None if omit_generative_model \
                            else model["generative_model"],
            generative_model_kwargs=generative_kwargs,
            callbacks_list=forwarded_callbacks,
            generative_callbacks_list=generative_forwarded_callbacks,
            return_details=True,
            return_features=continual_return_features,
            max_train_samples=continual_max_train_samples,
            max_val_samples=continual_max_val_samples,
            shuffle_buffer=continual_shuffle_buffer,
            pad=continual_pad,
            seed=seed,
            dtype_policy=dtype_policy,
            deterministic_ops=deterministic_ops,
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
        # Export task/class/phase metrics after all task fits have completed.
        if use_tensorboard:
            write_continual_tensorboard_summaries(
                details,
                tensorboard_path
            )

        history = {"continual_accuracy": details["accuracies"]}
        # Preserve optional ensemble scores beside ordinary continual accuracy.
        if details["ensemble_accuracies"]:
            history["continual_ensemble_accuracy"] = details[
                "ensemble_accuracies"
            ]

        # Use the ensemble validation matrix when selected and the ordinary matrix otherwise.
        selected_validation_matrix = (
            details.get("validation_ensemble_accuracy_matrix", [])
            if details.get("use_ensemble_accuracy", False)
            else details.get("validation_accuracy_matrix", [])
        )
        final_val_accuracy = [
            observed_mean(np.asarray(row)[:index + 1])
            for index, row in enumerate(selected_validation_matrix)
        ]

        # Fall back to task histories when no validation matrix is available.
        if not final_val_accuracy:
            # For each task, take the first available validation-accuracy history metric.
            final_val_accuracy = [
                next((
                    task_history[name][-1]
                    for name in (
                        "val_total_accuracy",
                        "val_classifier_accuracy",
                        "val_cls_token_accuracy",
                        "val_avg_pooling_accuracy",
                        "val_accuracy"
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
            **fit_call_kwargs
        ).history
        discriminator_history = model.fit_discriminator(
            x=trainset,
            epochs=epochs,
            validation_data=valset,
            callbacks=callbacks_list,
            verbose=training_verbose,
            **discriminator_kwargs
        ).history
        history = model._merge_result_dicts(
            (generator_history, discriminator_history),
            ("generator", "discriminator")
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
            method_input = trainset
            # VAE train calls need explicit batching and runtime seed arguments.
            if isinstance(model, VariationalAutoencoder):
                method_kwargs.setdefault("batch_size", batch_size)
                method_kwargs.setdefault("seed", seed)
                # Forward a caller-selected VAE shuffle buffer when one was supplied.
                if continual_shuffle_buffer is not None:
                    method_kwargs.setdefault(
                        "shuffle_buffer", continual_shuffle_buffer
                    )
            # The VAE resampling API operates on rows, while orchestration
            # supplies an already-batched dataset. Materialize those finite
            # batches once so ``train_num`` still samples individual examples.
            if isinstance(model, VariationalAutoencoder) and isinstance(
                trainset, tf.data.Dataset
            ):
                batches = [
                    tf.keras.utils.unpack_x_y_sample_weight(batch)
                    for batch in trainset.as_numpy_iterator()
                ]
                # An empty dataset cannot supply the arrays required by the VAE train API.
                if not batches:
                    raise ValueError("Training dataset must contain at least one batch.")

                method_input = np.concatenate([
                    np.asarray(batch_x) for batch_x, _, _ in batches
                ], axis=0)
                # Conditioned VAEs need dataset labels when explicit y was not supplied.
                if model.conditioned and "y" not in method_kwargs:
                    labels = [batch_y for _, batch_y, _ in batches]
                    # Concatenate complete batch labels; keep y absent if any batch lacks
                    # labels.
                    method_kwargs["y"] = None if any(
                        label is None for label in labels
                    ) else np.concatenate(labels, axis=0)

            trained = method(
                method_input,
                **method_kwargs
            )
        # Progressive methods own their stage and final epoch budgets.
        elif progressive_fit:
            trained = method(
                **fit_call_kwargs
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
        # Diffusion wrappers use TensorFlow weight prefixes; other models use HDF5 filenames.
        weights_name = "model.weights" if isinstance(
            model, DiffusionModel
        ) else "model.weights.h5"
        weights_path = os.path.join(
            image_callback.results_path,
            weights_name,
        )

        # Inspect the continual generator for topology changes; ordinary runs inspect the
        # model itself.
        checkpoint_model = model.get("generative_model") \
            if is_continual else model
        # Store the final progressive network topology before saving weights.
        if progressive_fit and config is not None and isinstance(
            checkpoint_model, DiffusionModel
        ):
            network_config = dict(_plain_config_value(
                checkpoint_model.network.get_config()
            ))
            # An unnamed configured network defaults to DiTClassifier with a classifier, or
            # DiffusionTransformer without one.
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
            # Resolve an unnamed dynamic network to the classifier or generator DiT
            # configuration section.
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
                # Save wrapper metadata in the matching V2, V1 classifier, or generator-only
                # config section.
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
            # Replay diffusion models use TensorFlow weight prefixes; replay VAEs use HDF5
            # filenames.
            replay_weights_name = "replay-model.weights" if isinstance(
                model["generative_model"], DiffusionModel
            ) else "replay-model.weights.h5"
            replay_weights_path = os.path.join(
                image_callback.results_path,
                replay_weights_name,
            )

            # Persist replay-model weights when generative replay was used.
            if model["generative_model"] is not None:
                model["generative_model"].save_weights(replay_weights_path)
                # Record both continual artifact paths in configuration metadata.
                if config is not None:
                    config.model.weights_path = replay_weights_path
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


def _report_final_visuals(
    model: tf.keras.Model | None,
    dataset_name: str,
    results_path: str | os.PathLike[str] | None,
    show_final_images: bool,
    save_final_images: bool,
    save_final_gifs: bool,
    final_images_steps: int,
    final_images_cfg_scale: float,
    seed: int | None
) -> None:
    """Generate requested final VAE images or diffusion images/trajectories.

    Unsupported/non-generative models and fully disabled output switches return
    without sampling. VAE output supports images: conditional models use their
    seen classes and unconditional models use the dataset class count. Flat VAE
    samples are reshaped/displayed only when their width matches the dataset image
    shape. Diffusion output can request final images and/or GIF trajectories using
    the wrapper's configured test network. Derived seeds isolate final sampling
    from other report streams.

    Args:
        model (tf.keras.Model | None): Trained VAE or diffusion wrapper. None and
            other model families are no-ops rather than errors.
        dataset_name (str): Dataset identifier used to obtain VAE image shape and
            unconditional sample count.
        results_path (str | os.PathLike[str] | None): Existing artifact directory.
            None is permitted for display-only output; file saving requires a path.
        show_final_images (bool): Display the final image grid when available.
        save_final_images (bool): Write the final image grid beneath results_path.
        save_final_gifs (bool): Save diffusion x_t/x0 sampling trajectories as a GIF.
            VAEs do not produce trajectories, so this flag alone performs no VAE work.
        final_images_steps (int): Reverse diffusion sampling steps. Ignored by VAEs;
            validity against the trained horizon is enforced by the wrapper.
        final_images_cfg_scale (float): Diffusion classifier-free guidance scale,
            also encoded in output filenames. Ignored by VAEs.
        seed (int | None): Master report seed used to derive separate VAE/diffusion
            final-sampling seeds. None supplies no new seed override, leaving the
            model's generation method to use its own configured seed behavior.

    Returns:
        None: May sample the model, display plots, and write PNG/GIF files. Existing
        output names follow plotting/GIF writer behavior; no metric values are returned.

    Raises:
        TypeError: If file output is enabled without a filesystem path.
        ValueError: If diffusion GIFs are requested for ``swap_noise_image=True``
            or a dataset/sampling option is invalid.
        OSError: If a requested image or GIF cannot be written."""

    # Ordinary classifiers have no sample-generation reporting protocol.
    if not isinstance(model, (VariationalAutoencoder, DiffusionModel)):
        return
    # Avoid any sampling work when no compatible output was requested.
    if not any((show_final_images, save_final_images, save_final_gifs)):
        return

    # VAEs produce final samples but no iterative denoising GIF trajectory.
    if isinstance(model, VariationalAutoencoder):
        # Ignore a GIF-only request because a VAE has no denoising trajectory.
        if not (show_final_images or save_final_images):
            return

        # Require a result directory for saved VAE images; display-only images need no path.
        results_path = _normalize_results_path(
            results_path,
            ("final image saving",) if save_final_images else (),
        )

        final_seed = derive_seed(seed, "final_report", "vae_generation")
        class_num, image_shape, _ = get_dataset_spec(dataset_name)
        # Generate one example for each known conditional class.
        if model.conditioned:
            classes = model.seen_classes or list(range(model.class_num or 0))
            imgs, _ = model.generate(
                classes=classes,
                samples_per_class=1,
                seed=final_seed
            )
        # Generate one unconditional example per dataset class.
        else:
            imgs = model.generate(
                samples_per_class=class_num,
                seed=final_seed
            )

        imgs = np.asarray(imgs)
        expected_width = int(np.prod(image_shape))
        # Plot only flattened samples compatible with the configured dataset.
        if imgs.ndim == 2 and len(imgs) > 0 and imgs.shape[-1] == expected_width:
            imgs = imgs.reshape((-1, *image_shape))
            # Choose a VAE image file only when saving is enabled; display-only output has no
            # path.
            imgs_save_path = os.path.join(
                results_path,
                "final-images.png"
            ) if save_final_images else None
            plot_images(
                imgs,
                show_images=show_final_images,
                save_path=imgs_save_path
            )

        return

    # Swapped-noise wrappers cannot expose intermediate denoising frames.
    if save_final_gifs and model.swap_noise_image:
        raise ValueError(
            "Final GIF trajectories are unavailable when swap_noise_image=True."
        )

    diffusion_file_outputs = []
    # PNG output requires a destination independently of GIF output.
    if save_final_images:
        diffusion_file_outputs.append("final image saving")
    # A denoising animation is likewise a file-only output.
    if save_final_gifs:
        diffusion_file_outputs.append("final GIF saving")
    results_path = _normalize_results_path(results_path, diffusion_file_outputs)

    final_seed = derive_seed(seed, "final_report", "diffusion_sampling")

    # Sample and save the full denoising trajectory when GIF output is enabled.
    if save_final_gifs:
        imgs, frames1, frames2 = model.sample(
            network_name=model.test_network_name,
            scale=final_images_cfg_scale,
            steps=final_images_steps,
            return_x_ts=True,
            return_x0s=True,
            seed=final_seed
        )
        create_gif(
            os.path.join(
                results_path,
                f"final-gifs_steps-{final_images_steps}"
                f"_scale-{final_images_cfg_scale:.1f}.gif",
            ),
            frames1,
            frames2
        )
    # Sample only final images when a trajectory is unnecessary.
    else:
        imgs = model.sample(
            network_name=model.test_network_name,
            scale=final_images_cfg_scale,
            steps=final_images_steps,
            seed=final_seed
        )

    # Choose a diffusion image file only for enabled image saving.
    imgs_save_path = os.path.join(
        results_path,
        f"final-images_steps-{final_images_steps}"
        f"_scale-{final_images_cfg_scale:.1f}.png",
    ) if save_final_images else None

    # Render final samples only when display or PNG output was requested.
    if show_final_images or save_final_images:
        plot_images(
            imgs,
            show_images=show_final_images,
            save_path=imgs_save_path
        )


def report(
    config: Config | None = None,
    history: Mapping[str, Sequence[float]] | None = None,
    model: tf.keras.Model | dict[str, object] | None = None,
    trainset: tf.data.Dataset | Callable[..., object] | None = None,
    valset: tf.data.Dataset | None = None,
    **kwargs: object
) -> dict[str, object]:
    """Plot training history, evaluate models, and emit final metric/sample artifacts.

    Ordinary models can evaluate training and validation datasets. Diffusion models
    evaluate raw and, when enabled, EMA branches; V2 includes both optimization
    phases. Continual bundles reuse the learner's completed details and trajectories
    instead of rerunning its callable loader. Their validation metric mapping stays
    separate from ordinary/test summaries, including development runs without test
    observations. Optional CSV output also exports all five continual report tables.

    Args:
        config (Config | None): Typed report/training/dataset settings. Default None
            selects the direct keyword defaults in ``_resolve_reporting_options``.
        history (Mapping[str, Sequence[float]] | None): Metric trajectories. Default
            None skips ordinary history plotting. A continual bundle must supply
            ``continual_accuracy`` and may supply ``continual_ensemble_accuracy``.
        model (tf.keras.Model | dict[str, object] | None): Trained model or continual
            bundle. Default None is a signature placeholder; enabled evaluation
            requires the corresponding model. Bundles may include ``continual_details``.
        trainset (tf.data.Dataset | Callable[..., object] | None): Training dataset
            for optional ordinary evaluation. Default None; continual loaders are
            not invoked by this reporting function.
        valset (tf.data.Dataset | None): Optional validation data. Default None
            skips ordinary validation evaluation even if its report switch is True.
        **kwargs (object): Direct report switches/path/sampling settings described
            by ``_resolve_reporting_options``. They include history plots/CSV,
            train/validation evaluation, ``evaluate_ensemble_accuracy`` plus its
            option mapping, final image/GIF switches, steps, CFG scale, and seed.
            Typed Config settings take precedence when config is supplied.

    Returns:
        dict[str, object]: Ordinary ``trainset_eval``/``valset_eval`` dictionaries;
        diffusion ``<split>_network_eval`` and optional ``<split>_ema_eval`` results;
        or continual trajectory summaries, final metric dictionaries, selected
        schedule/matrix fields, and optional ensemble summaries. Disabled ordinary
        evaluations return an empty mapping. Plotting/writing and evaluation metric
        updates are side effects; no model training is performed.

    Raises:
        TypeError: If an enabled file output requires a missing artifact path.
        ValueError: If ensemble evaluation is incompatible with the selected
            ordinary model or final sampling/GIF controls are unsupported.
        OSError: If requested report files cannot be written.
        KeyError: If a continual history omits its required accuracy trajectory."""

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
    seed = reporting_options["seed"]
    is_continual = isinstance(model, dict)

    # Continual visual reports use the replay generator; ordinary reports use the supplied
    # model.
    visual_model = model.get("generative_model") if is_continual else model
    result_path_consumers = []
    # History and evaluation CSV paths are prepared before model reporting.
    if save_history_plot:
        result_path_consumers.append("history-plot saving")
    # Both ordinary and continual CSV reporters require the same artifact root.
    if save_csv:
        result_path_consumers.append("CSV saving")
    # Final PNGs are meaningful only for supported generative models.
    if save_final_images and isinstance(
        visual_model, (VariationalAutoencoder, DiffusionModel)
    ):
        result_path_consumers.append("final image saving")
    # Preserve the more specific swapped-noise error for unsupported GIFs.
    if save_final_gifs and isinstance(visual_model, DiffusionModel) \
    and not visual_model.swap_noise_image:
        result_path_consumers.append("final GIF saving")
    results_path = _normalize_results_path(results_path, result_path_consumers)

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
            "average_accuracy": observed_mean(history["continual_accuracy"]),
            "final_accuracy": float(history["continual_accuracy"][-1])
        }

        details = model.get("continual_details", {})
        # Expose task-balanced CL metrics and the complete evaluation matrix.
        eval_results.update(details.get("continual_metrics", {}))
        eval_results["validation_continual_metrics"] = details.get(
            "validation_continual_metrics", {}
        )
        for name in (
            "class_order",
            "task_classes",
            "accuracy_matrix",
            "new_task_accuracy",
            "old_task_accuracy",
            "seed"
        ):
            # Copy only schedule metadata recorded by the continual learner.
            if name in details:
                eval_results[name] = details[name]

        # Add ensemble summaries when the continual learner produced them.
        if "continual_ensemble_accuracy" in history \
        and len(history["continual_ensemble_accuracy"]) > 0:
            eval_results.update({
                "average_ensemble_accuracy": observed_mean(
                    history["continual_ensemble_accuracy"]
                ),
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
            write_continual_csv_artifacts(
                details,
                results_path,
                metadata={
                    "dtype_policy": details.get("dtype_policy"),
                    "seed": details.get("seed"),
                    "snapshot_network_name": details.get(
                        "snapshot_network_name"
                    ),
                    "use_ensemble_accuracy": details.get(
                        "use_ensemble_accuracy", False
                    )
                }
            )

        # Continual bundles still honor final visual reporting for their
        # trained replay model; classifier-only runs are ignored safely.
        _report_final_visuals(
            model.get("generative_model"),
            dataset_name,
            results_path,
            show_final_images,
            save_final_images,
            save_final_gifs,
            final_images_steps,
            final_images_cfg_scale,
            seed
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

        _report_final_visuals(
            model,
            dataset_name,
            results_path,
            show_final_images,
            save_final_images,
            save_final_gifs,
            final_images_steps,
            final_images_cfg_scale,
            seed
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
                    ensemble_accuracy_kwargs
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

    _report_final_visuals(
        model,
        dataset_name,
        results_path,
        show_final_images,
        save_final_images,
        save_final_gifs,
        final_images_steps,
        final_images_cfg_scale,
        seed
    )

    return eval_results


def main(
    config: Config | None = None,
    teacher_network: tf.keras.Model | None = None,
    **kwargs: object
) -> dict[str, object]:
    """Run the complete Config/direct-mode training pipeline and return its artifacts.

    Normalize the task first, install the effective seed and numeric policy, obtain
    datasets, construct the model, train it, and report into the same concrete run
    directory. Continual datasets are loader callables and models are bundles;
    ordinary tasks use prepared datasets and a model. Training may materialize
    schedule/topology/weight paths back into a supplied Config.

    Args:
        config (Config | None): Complete typed project settings. Default None uses
            kwargs throughout the pipeline. Its training task is normalized in place
            before dataset/model construction; other resolved fields can be updated
            by training.
        teacher_network (tf.keras.Model | None): Optional independent runtime teacher
            forwarded to model construction for supported diffusion distillation.
            Default None. Live teacher objects are not embedded in config YAML;
            continual self-distillation can create later-task teachers automatically.
        **kwargs (object): Direct dataset/model/training/reporting settings. Task
            defaults to ``"legacy"``; continual calls use ``task="continual"``
            and ``continually_learn_kwargs``. Direct runtime defaults include
            ``dtype_policy="float32"`` and ``deterministic_ops=False``; effective
            seed behavior is delegated to the runtime task contract. The factory,
            training, and reporting functions document their remaining defaults.

    Returns:
        dict[str, object]: ``model`` (trained model or updated bundle), ``history``
        (epoch/task metrics), ``evaluations`` (final report mapping), and
        ``results_path`` (the resolved artifact directory, possibly None for direct
        calls with every file consumer disabled). Models remain live objects.
        The call can load data/weights, set process-wide runtime state, train,
        display plots, and write the configured artifacts.

    Raises:
        TypeError: If the task is not a string or a downstream API receives an
            unsupported parameter/path type.
        ValueError: If task normalization or dataset/model/training validation fails.
        OSError: If requested data/weight reads or artifact writes fail. Exceptions
            from model training and reporting otherwise propagate without rollback."""

    # Revalidate mutable typed configs at the complete-pipeline boundary.
    if config is not None:
        config.training.task = normalize_training_task(config.training.task)
    # Canonicalize direct task selection before dataset/model side effects.
    else:
        kwargs["task"] = normalize_training_task(kwargs.get("task", "legacy"))

    # Announce typed configuration and obtain its seed.
    if config is not None:
        print(
            "Initiating training process with "
            f"the following settings:\n{config}"
        )

        seed = effective_seed(config)
        dtype_policy = config.training.dtype_policy
        deterministic_ops = config.training.deterministic_ops
    # Announce direct settings and obtain their seed.
    else:
        print(
            "Initiating training process with "
            f"the following settings:\n{kwargs}"
        )

        seed = effective_seed(
            seed=kwargs.get("seed"),
            task=kwargs.get("task")
        )
        dtype_policy = kwargs.get("dtype_policy", "float32")
        deterministic_ops = kwargs.get("deterministic_ops", False)

    # Install policy and seed before constructing datasets, models, layers, or
    # optimizers. Keras seeds Python, NumPy, and TensorFlow together.
    configure_runtime(seed, dtype_policy, deterministic_ops)

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

    run_state = {}
    history = train_model(
        config,
        model,
        trainset,
        valset=valset,
        _run_state=run_state,
        **kwargs
    )

    # Direct calls have no mutable Config through which training can publish
    # the timestamped child used for every artifact.
    if config is None:
        kwargs["results_path"] = run_state["results_path"]

    evaluations = report(
        config,
        history,
        model,
        trainset,
        valset=valset,
        **kwargs
    )

    # Return the resolved Config path when configured and the resolved direct-mode path
    # otherwise.
    return {
        "model": model,
        "history": history,
        "evaluations": evaluations,
        "results_path": config.training.results_path if config is not None
                        else kwargs["results_path"]
    }


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    config = load_config()
    main(config)
