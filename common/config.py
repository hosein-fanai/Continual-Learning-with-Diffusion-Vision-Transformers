"""Dataclass configuration tree and YAML serialization for experiments.

Legacy DiT sections expose :meth:`KwargsMixin.kwargs`. Generic model families
use the compact ``name``/``kwargs`` fields consumed by :mod:`common.train` and
the shared HPO runner. ``ContinuallyLearnConfig`` contains the policy unique to
class-incremental runs while the other sections remain reusable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, TypeVar
from collections.abc import Mapping

import os

import yaml


SectionT = TypeVar("SectionT")


def _all_depth_ids() -> list[int | None]:
    """Return the default marker selecting every eligible network depth.

    Returns:
        list[int | None]: A new ``[None]`` list for one dataclass instance.
    """

    return [None]


def _default_regularizer_range() -> dict[str, int]:
    """Return the default normalized regularizer interval.

    Returns:
        dict[str, int]: A new ``{"start": 0, "end": 1}`` mapping.
    """

    return {"start": 0, "end": 1}


def _default_feature_aggregation() -> dict[int, list[int]]:
    """Return the default final-depth feature route.

    Returns:
        dict[int, list[int]]: A new mapping from classifier depth 1 to the
        source model's final depth.
    """

    return {1: [-1]}


def _default_classifier_connection() -> dict[int, list[int]]:
    """Return the default route from the final source classifier depth.

    Returns:
        dict[int, list[int]]: A new ``{-1: [-1]}`` mapping.
    """

    return {-1: [-1]}


def _default_vae_hiddens() -> list[int]:
    """Return the default single dense VAE hidden width.

    Returns:
        list[int]: A new ``[16]`` list.
    """

    return [16]


def _default_unet_widths() -> list[int]:
    """Return the standard three-stage U-Net widths.

    Returns:
        list[int]: A new ``[32, 64, 96]`` list.
    """

    return [32, 64, 96]


def _default_buffer_kwargs() -> dict[str, object]:
    """Return default fixed replay-buffer controls.

    Returns:
        dict[str, object]: New capacity, sampling, insertion, and seed values.
    """

    return {
        "maxlen": 10_000, 
        "sample_num": 1_000, 
        "insert_num": 1_000, 
        "seed": None
    }


def _default_generative_replay_kwargs() -> dict[str, int]:
    """Return default generative-replay sample counts.

    Returns:
        dict[str, int]: New training and per-class generation counts.
    """

    return {"train_num": 1_000, "samples_per_class": 1_000}


class KwargsMixin:
    """Convert a configuration dataclass into constructor keyword arguments."""

    def kwargs(self: KwargsMixin) -> dict[str, Any]:
        """Return a recursively copied dictionary of all dataclass fields.

        Returns:
            dict[str, Any]: Every field, including values equal to defaults and
            ``None``.  Keys are not renamed or filtered before callers expand
            the result with ``**``.
        """

        return asdict(self)


@dataclass
class DiffusionTransformerConfig(KwargsMixin):
    """Arguments forwarded to ``DiffusionTransformer``."""

    num_classes: int | None = 10
    use_cfg: bool = True
    timesteps: int = 1_000
    image_size: int = 28
    channels: int = 1
    patch_size: int = 2
    dim: int = 32
    dim_forced: bool = True
    patchify_with_cnn: bool = False
    patches_pos_embed_type: str = "2d_sincos"
    patches_pos_merger_type: str = "add"
    patches_conds_merger_type: str | None = None
    shift_inputs: bool = False
    cond_dim: int | None = None
    cond_type: str | None = "time_label"
    conds_merger_type: str = "add"
    time_embed_type: str = "1d_sincos"
    time_freq_dim: int | None = None
    time_embed_trainable: bool = False
    time_mlp_ratio: float | None = None
    label_embed_type: str = "new_weight"
    label_embed_trainable: bool = False
    label_freq_dim: int | None = None
    label_mlp_ratio: float | None = None
    cls_token_type: str | None = None
    cls_token_freq_dim: int | None = None
    cls_token_mlp_ratio: float | None = None
    cls_token_pos_merger_type: str = "add"
    depth: int = 2
    connection_ids_dict: dict = field(default_factory=dict)
    connection_kwargs: dict = field(default_factory=dict)
    cross_attention_ids_dict: dict = field(default_factory=dict)
    cross_attention_kwargs: dict = field(default_factory=dict)
    cross_attention_plug_type: str = "values"
    vit_block_ids: list[int | None] = field(default_factory=_all_depth_ids)
    use_decoder_ids: list[int | None] = field(default_factory=list)
    mha_key_dim: int | None = None
    mha_value_dim: int | None = None
    mha_num_heads: int = 4
    vit_block_mlp_ratio: float = 4.0
    vit_block_mlp_output_dims: dict[int, int] = field(default_factory=dict)
    ln_mlp_ratio: float | None = None
    ln_no_adaptation: bool = False
    drop_prob: float = 0.0
    drop_per_sample: bool = True
    local_mixer_ids: list[int | None] = field(default_factory=list)
    local_mixer_kwargs: dict = field(default_factory=dict)
    downsample_ids: list[int | None] = field(default_factory=list)
    downsample_kwargs: dict = field(default_factory=dict)
    upsample_ids: list[int | None] = field(default_factory=list)
    upsample_kwargs: dict = field(default_factory=dict)
    reshaper_ids_dict: dict[int, str] = field(default_factory=dict)
    reshaper_kwargs: dict = field(default_factory=dict)
    cls_token_regularizer_ids: list[int | None] = field(default_factory=list)
    cls_token_regularizer_kwargs: dict = field(
        default_factory=_default_regularizer_range
    )
    final_ffn_activation_func: str = "linear"
    use_refiner_cnn: bool = False
    refiner_cnn_hidden_dim: int | None = None
    refiner_cnn_residual: bool = True
    final_activation_func: str = "linear"
    use_unpatchify: bool = True
    name_prefix: str = ""
    build: bool = True


@dataclass
class DiTClassifierConfig(DiffusionTransformerConfig):
    """Arguments forwarded to ``DiTClassifier``, including inherited ones."""

    aggregate_from_noises: bool = False
    feature_aggregation_ids_dict: dict = field(
        default_factory=_default_feature_aggregation
    )
    feature_aggregation_kwargs: dict = field(default_factory=dict)
    cross_attention_aggregation_ids_dict: dict = field(default_factory=dict)
    cross_attention_aggregation_kwargs: dict = field(default_factory=dict)
    classifier_only_cls_token: bool = True
    clf_dim: int | None = None
    clf_dim_forced: bool = False
    clf_cond_type: str | None = "time_label"
    clf_cls_token_type: str | None = "new_weight"
    clf_depth: int = 1
    clf_connection_ids_dict: dict = field(
        default_factory=_default_classifier_connection
    )
    clf_connection_kwargs: dict | None = None
    clf_cross_attention_ids_dict: dict = field(default_factory=dict)
    clf_cross_attention_kwargs: dict | None = None
    clf_cross_attention_plug_type: str | None = None
    clf_vit_block_ids: list[int | None] = field(default_factory=_all_depth_ids)
    clf_use_decoder_ids: list[int | None] = field(default_factory=list)
    clf_mha_key_dim: int | None = None
    clf_mha_value_dim: int | None = None
    clf_mha_num_heads: int | None = None
    clf_vit_block_mlp_ratio: float | None = None
    clf_vit_block_mlp_output_dims: dict[int, int] | None = None
    clf_ln_mlp_ratio: float | None = None
    clf_ln_no_adaptation: bool | None = None
    clf_drop_prob: float | None = None
    clf_drop_per_sample: bool | None = None
    clf_local_mixer_ids: list[int | None] = field(default_factory=list)
    clf_local_mixer_kwargs: dict | None = None
    clf_downsample_ids: list[int | None] = field(default_factory=list)
    clf_downsample_kwargs: dict | None = None
    clf_upsample_ids: list[int | None] = field(default_factory=list)
    clf_upsample_kwargs: dict | None = None
    clf_reshaper_ids_dict: dict[int, str] = field(default_factory=dict)
    clf_reshaper_kwargs: dict = field(default_factory=dict)
    clf_cls_token_regularizer_ids: list[int | None] = field(default_factory=list)
    clf_cls_token_regularizer_kwargs: dict | None = None
    force_global_avg_pooling: bool = False
    classifier_mlp_ratio: int | None = None
    classifier_mlp_activation_func: str = "tanh"
    dropout_rate: float = 0.0


@dataclass
class DiTDecoderConfig(DiffusionTransformerConfig):
    """Arguments forwarded to ``DiTDecoder``."""

    encoder_output_grid_size: int | None = None
    encoder_output_dim: int | None = None
    encoder_feature_grid_sizes: list[int | None] | None = None
    encoder_feature_dims: list[int] | None = None
    shift_inputs: bool = True
    use_decoder_ids: list[int | None] = field(default_factory=_all_depth_ids)
    decoder_separate_cond: bool = False
    use_causal_mask: bool = True
    feature_aggregation_ids_dict: dict = field(default_factory=dict)
    feature_aggregation_kwargs: dict = field(default_factory=dict)
    cross_attention_aggregation_ids_dict: dict = field(default_factory=dict)
    cross_attention_aggregation_kwargs: dict = field(default_factory=dict)


@dataclass
class DiTEncoderDecoderConfig(DiffusionTransformerConfig):
    """Arguments forwarded to ``DiTEncoderDecoder``."""

    encoder_kwargs: dict[str, object] | None = None
    decoder_kwargs: dict[str, object] | None = None


@dataclass
class DiTEncoderDecoderClassifierConfig(DiTClassifierConfig):
    """Arguments forwarded to ``DiTEncoderDecoderClassifier``."""

    encoder_kwargs: dict[str, object] | None = None
    decoder_kwargs: dict[str, object] | None = None


@dataclass
class UNetConfig(KwargsMixin):
    """Arguments forwarded to ``UNet``."""

    num_classes: int | None = 10
    use_cfg: bool = True
    timesteps: int = 1_000
    image_size: int = 32
    channels: int = 1
    widths: list[int] = field(default_factory=_default_unet_widths)
    block_depth: int = 2
    bottleneck_width: int = 128
    bottleneck_depth: int = 2
    image_embedding_dim: int = 21
    time_embedding_dim: int = 22
    label_embedding_dim: int = 21
    activation_func: str = "swish"
    final_activation_func: str = "linear"
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
    downsampling_method: str = "avg_pooling"
    upsampling_method: str = "interpolate"
    upsampling_interpolation: str = "bilinear"
    use_skip_connections: bool | None = None
    reshaper_ids_dict: dict[int, str] = field(default_factory=dict)
    reshaper_kwargs: dict = field(default_factory=dict)
    cls_token_regularizer_ids: list[int | None] = field(default_factory=list)
    cls_token_regularizer_kwargs: dict = field(
        default_factory=_default_regularizer_range
    )
    extra_depth_specs: list[object] = field(default_factory=list)
    name_prefix: str = ""
    build: bool = True


@dataclass
class UNetClassifierConfig(UNetConfig):
    """Arguments forwarded to ``UNetClassifier``."""

    aggregate_from_noises: bool = False
    feature_aggregation_ids_dict: dict = field(
        default_factory=_default_feature_aggregation
    )
    classifier_only_cls_token: bool = False
    clf_dim: int | None = None
    clf_depth: int = 1
    clf_block_depth: int = 1
    clf_reshaper_kwargs: dict = field(default_factory=dict)
    clf_cls_token_regularizer_ids: list[int | None] = field(default_factory=list)
    force_global_avg_pooling: bool = True
    classifier_mlp_ratio: float | None = None
    classifier_mlp_activation_func: str = "tanh"


@dataclass
class DiffusionModelConfig(KwargsMixin):
    """Arguments forwarded to ``DiffusionModel`` except ``network``.

    ``seen_classes`` is normally empty in input configs. Dynamic diffusion
    checkpoint saving fills it with the live dataset-label-to-zero-based-target
    mapping. On a continual reload, the raw constructor still receives
    ``num_classes=None``
    and the wrapper uses this mapping to restore the grown topology.
    """

    use_ema: bool = True
    test_network_name: str = "ema"
    ema_decay: float = 0.999
    scheduler_name: str = "clipped_cosine"
    modify_first_t: bool = False
    p_uncond: float = 0.1
    train_cfg_scale: float | None = None
    test_cfg_scale: float = 4.0
    test_steps: int = 50
    test_eta: float = 0.0
    noise_loss_coef: float = 1.0
    image_loss_coef: float = 0.0
    kl_loss_coef: float = 0.0
    ctr_loss_coef: float = 0.0
    kl_train_type: str = "cond"
    ctr_train_type: str = "cond"
    train_noisified_min_timesteps: int = 0
    train_noisified_max_timesteps: int | None = None
    test_noisified_min_timesteps: int = 0
    test_noisified_max_timesteps: int | None = None
    resize_method: str = "area"
    resize_antialias: bool = True
    swap_noise_image: bool = False
    seen_classes: dict[object, int] = field(default_factory=dict)
    seed: int | None = None


@dataclass
class DiffusionClassifierConfig(DiffusionModelConfig):
    """Arguments forwarded to ``DiffusionClassifier`` except ``network``."""

    mask_by_nulls: bool = True
    mask_by_t_threshold: bool = False
    mask_t_percentage: int = 70
    use_ensemble_loss_instead: bool = False
    clf_train_type: str = "cond"
    clf_loss_coef: float = 8.6e-3


@dataclass
class DiffusionClassifierV2Config(DiffusionClassifierConfig):
    """Arguments forwarded to ``DiffusionClassifierV2`` except ``network``."""

    clf_loss_coef: float = 1.0
    clf_vars_embedding_ids: list[int] = field(default_factory=list)
    clf_vars_noise_part_ids: list[int] = field(default_factory=list)
    clf_train_noisified_max_timesteps: int | None = None
    clf_test_noisified_max_timesteps: int | None = None


@dataclass
class VariationalAutoencoderConfig(KwargsMixin):
    """Arguments forwarded to ``VariationalAutoencoder``."""

    data_dim: int = 2_048
    latent_dim: int = 8
    hiddens_dims: list[int] = field(default_factory=_default_vae_hiddens)
    hiddens_kwargs: dict = field(default_factory=dict)
    last_activation: str | None = "tanh"
    beta: float = 0.25
    conditioned: bool = False
    class_num: int | None = None
    compile: bool = True
    compile_args: dict = field(default_factory=dict)


@dataclass
class VAEClassifierConfig(KwargsMixin):
    """Arguments forwarded to ``VAEClassifier`` except classifier inputs."""

    data_dim: int = 2_048
    latent_dim: int = 8
    hiddens_dims: list[int] = field(default_factory=_default_vae_hiddens)
    hiddens_kwargs: dict = field(default_factory=dict)
    last_activation: str | None = "tanh"
    beta: float = 0.25
    alpha: float = 1.0
    compile_args: dict = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Dataset selection, preprocessing, batching, and optional trial limits.

    Attributes:
        name (str): ``"mnist"``, ``"fmnist"``, ``"cifar10"``, or
            ``"cifar100"``.
        preprocess (str | None): ``"min-max"``, ``"normalize"``,
            ``"standardize"``/``"diffusion"``, or no scaling.
        indices (list[int] | None): Class IDs retained by ordinary dataset
            construction. The continual loop introduces the zero-based prefix
            selected by ``continually_learn.class_num``.
        validation_ratio (float): Fraction of training rows reserved for a
            stratified validation split; ``0`` disables the split.
        features_path (str | None): Base path of an optional saved feature
            archive.
        return_features (bool): Use saved features instead of raw images.
        onehot_labels (bool): Return full-width categorical labels instead of
            sparse IDs.
        max_train_samples (int | None): Optional positive training-row limit.
            Continual runs apply a class-preserving limit once before per-task
            selection, so the value must cover every represented class.
        max_val_samples (int | None): Optional positive evaluation-row limit.
            Limits preserve represented classes. Continual runs limit
            validation rows, or test rows when no validation split exists.
        batch_size (int): Positive examples per batch; defaults to 128.
        shuffle_buffer (int): Training shuffle-buffer capacity, including each
            continual task. Values above zero enable shuffling; ``0`` disables
            it. Defaults to 10,000.
        pad (int): Symmetric zero-padding applied to raw images. Continual runs
            apply it before replay; saved features and pretrained/hp-tuned
            classifiers do not support it.
        trainset_len (int | None): Number of prepared training batches,
            populated by ``get_datasets`` for optimizer scheduling.
    """

    name: str = "mnist"
    preprocess: str | None = None
    indices: list[int] | None = None
    validation_ratio: float = 0.
    features_path: str | None = None
    return_features: bool = False
    onehot_labels: bool = False
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    batch_size: int = 128
    shuffle_buffer: int = 10_000
    pad: int = 0
    trainset_len: int | None = None


@dataclass
class ModelConfig:
    """Select the raw network, wrapper, visibility, and initial weights.

    Attributes:
        name (str | None): Generic raw-model/family name. ``None`` retains the
            legacy compact DiT selection below.
        kwargs (dict): Generic raw-model or classifier constructor arguments.
            When nonempty, these retain precedence over the typed section for
            ``name``. Continual diffusion construction overrides
            ``num_classes`` with ``None``.
        wrapper_kwargs (dict): Generic diffusion-wrapper arguments.
        classifier_name (str | None): Target classifier for continual runs.
        classifier_kwargs (dict): Target-classifier architecture arguments.
        loss_function (str): Generative reconstruction/noise loss passed to
            Keras compilation; ``"mse"`` by default.
        with_classifier (bool): Build ``DiTClassifier`` inside
            ``DiffusionClassifier`` when true; otherwise build
            ``DiffusionTransformer`` inside ``DiffusionModel``.
        show_network_summary (bool): Print the wrapper/network summary after
            construction.
        weights_path (str | None): Keras weights file loaded after construction,
            or ``None`` for fresh weights. In continual runs it initializes a
            VAE replay model, or the classifier and its incremental head
            prefixes for classifier-only and buffer-based runs. A continual
            diffusion checkpoint requires a paired config containing the
            current ``num_classes`` and zero-based wrapper ``seen_classes``
            mapping. The
            continual factory still constructs a dynamic raw network and the
            wrapper grows it before loading these weights. Training updates
            this field to its saved ``model.weights.h5`` path.
        diffusion_transformer (DiffusionTransformerConfig): Raw denoising
            network settings used only when ``with_classifier=False``.
        dit_classifier (DiTClassifierConfig): Raw joint network settings used
            only when ``with_classifier=True``.
        diffusion_model (DiffusionModelConfig): Process-wrapper settings used
            only when ``with_classifier=False``.
        diffusion_classifier (DiffusionClassifierConfig): Classifier wrapper
            settings used only when ``with_classifier=True``.
        Other typed fields provide constructor settings for their matching
            ``name`` or ``wrapper_name`` selection.
    """

    with_classifier: bool = True
    show_network_summary: bool = True
    weights_path: str | None = None
    loss_function: str = "mse"

    # Generic selections used by config-driven experiments and HPO. ``None``
    # keeps the original DiT/DiT-classifier path above fully backward compatible.
    name: str | None = None
    wrapper_name: str | None = None
    kwargs: dict = field(default_factory=dict)
    wrapper_kwargs: dict = field(default_factory=dict)
    classifier_name: str | None = None
    classifier_kwargs: dict = field(default_factory=dict)

    diffusion_transformer: DiffusionTransformerConfig = field(
        default_factory=DiffusionTransformerConfig
    )
    dit_classifier: DiTClassifierConfig = field(
        default_factory=DiTClassifierConfig
    )
    dit_decoder: DiTDecoderConfig = field(default_factory=DiTDecoderConfig)
    dit_encoder_decoder: DiTEncoderDecoderConfig = field(
        default_factory=DiTEncoderDecoderConfig
    )
    dit_encoder_decoder_classifier: DiTEncoderDecoderClassifierConfig = field(
        default_factory=DiTEncoderDecoderClassifierConfig
    )
    unet: UNetConfig = field(default_factory=UNetConfig)
    unet_classifier: UNetClassifierConfig = field(
        default_factory=UNetClassifierConfig
    )
    variational_autoencoder: VariationalAutoencoderConfig = field(
        default_factory=VariationalAutoencoderConfig
    )
    vae_classifier: VAEClassifierConfig = field(
        default_factory=VAEClassifierConfig
    )
    diffusion_model: DiffusionModelConfig = field(
        default_factory=DiffusionModelConfig
    )
    diffusion_classifier: DiffusionClassifierConfig = field(
        default_factory=DiffusionClassifierConfig
    )
    diffusion_classifier_v2: DiffusionClassifierV2Config = field(
        default_factory=DiffusionClassifierV2Config
    )

    def __post_init__(self: ModelConfig) -> None:
        """Convert nested model-section mappings to typed dataclass instances.

        Returns:
            None: The nested attributes are normalized in place.  Existing
            instances are preserved; mappings are expanded as constructor
            keywords and may omit fields to use their defaults.
        """

        self.diffusion_transformer = _section(self.diffusion_transformer, DiffusionTransformerConfig)
        self.dit_classifier = _section(self.dit_classifier, DiTClassifierConfig)
        self.dit_decoder = _section(self.dit_decoder, DiTDecoderConfig)
        self.dit_encoder_decoder = _section(
            self.dit_encoder_decoder, DiTEncoderDecoderConfig
        )
        self.dit_encoder_decoder_classifier = _section(
            self.dit_encoder_decoder_classifier,
            DiTEncoderDecoderClassifierConfig,
        )
        self.unet = _section(self.unet, UNetConfig)
        self.unet_classifier = _section(
            self.unet_classifier, UNetClassifierConfig
        )
        self.variational_autoencoder = _section(
            self.variational_autoencoder, VariationalAutoencoderConfig
        )
        self.vae_classifier = _section(
            self.vae_classifier, VAEClassifierConfig
        )
        self.diffusion_model = _section(self.diffusion_model, DiffusionModelConfig)
        self.diffusion_classifier = _section(self.diffusion_classifier, DiffusionClassifierConfig)
        self.diffusion_classifier_v2 = _section(
            self.diffusion_classifier_v2, DiffusionClassifierV2Config
        )


@dataclass
class OptimizerConfig:
    """Optimizer and learning-rate schedule settings.

    Attributes:
        initial_learning_rate (float): Initial cosine-decay learning rate;
            defaults to ``5e-3``.
        decay_steps (int | None): Positive cosine-decay duration in optimizer
            steps.  ``None`` is replaced during model construction with
            ``epochs * trainset_len``.
        name (str): ``"adam"``, ``"adamw"``, ``"nadam"``, ``"rmsprop"``,
            or ``"sgd"``.
        schedule (str | None): ``"cosine"``, ``"constant"``, or ``None``.
        weight_decay (float | None): Optional AdamW-style weight decay; nonzero
            values require ``name="adamw"`` under TensorFlow 2.10.
        momentum (float): Momentum used by RMSprop/SGD.
        clipnorm (float | None): Optional global gradient-norm clipping value.
    """

    initial_learning_rate: float = 5e-3
    decay_steps: int | None = None
    name: str = "adam"
    schedule: str = "cosine"
    weight_decay: float | None = None
    momentum: float = 0.0
    clipnorm: float | None = None


@dataclass
class ContinuallyLearnConfig(KwargsMixin):
    """Class-incremental learning and replay settings.

    Dataset selection and preprocessing remain in :class:`DatasetConfig`,
    model construction remains in :class:`ModelConfig`, and fit/reporting
    controls remain in :class:`TrainingConfig` and :class:`ReportingConfig`.
    This section contains only settings specific to the continual loop.

    Attributes:
        class_num (int | None): Number of classes introduced across the run.
            ``None`` uses the selected dataset's complete class count; an
            explicit value must be between 2 and that count. This controls the
            task sequence, not the dynamic diffusion model's initial head width.
        remove_prev_classes (bool): Train later tasks on only the newly
            introduced class when true; otherwise train on every seen class.
        keep_same_model (bool): Carry shared classifier weights and old output
            columns into the next expanded classifier.
        use_loaded_opt (bool): Reuse the optimizer stored with the configured
            classifier template. This preserves the optimizer created by
            :func:`common.model.get_model` by default.
        use_buffer (bool): Use fixed-size sample replay instead of a generative
            replay model.
        buffer_kwargs (dict): ``ReplayBuffer`` controls ``maxlen``,
            ``sample_num``, ``insert_num``, and ``seed``.
        plot_results (bool): Plot accuracy against the number of seen classes.
        generative_model_kwargs (dict): Replay-model controls ``train_num``
            and ``samples_per_class``.
        use_generative_model_classifier (bool): Use the classifier attached to
            a VAE or diffusion replay model instead of the standalone model.
        train_classifier_separately (bool): Add the separate classifier phase
            required by ``DiffusionClassifierV2`` and optionally used by a
            ``VAEClassifier``.
        evaluate_ensemble_accuracy (bool): Also evaluate diffusion-classifier
            ensemble accuracy after each continual task.
        ensemble_accuracy_kwargs (dict): Options forwarded to
            ``DiffusionClassifier.evaluate_ensemble_accuracy``.
        return_details (bool): Return task histories and final models together
            with accuracies from a direct configured call.
    """

    class_num: int | None = None
    remove_prev_classes: bool = True
    keep_same_model: bool = True
    use_loaded_opt: bool = True
    use_buffer: bool = False
    buffer_kwargs: dict[str, object] = field(
        default_factory=_default_buffer_kwargs
    )
    plot_results: bool = True
    generative_model_kwargs: dict[str, int] = field(
        default_factory=_default_generative_replay_kwargs
    )
    use_generative_model_classifier: bool = False
    train_classifier_separately: bool = False
    evaluate_ensemble_accuracy: bool = False
    ensemble_accuracy_kwargs: dict[str, object] = field(default_factory=dict)
    return_details: bool = False


@dataclass
class TrainingConfig:
    """Task, fit-loop, callback, TensorBoard, and persistence settings.

    Attributes:
        project_tag (str | None): Optional result-run identifier passed to the
            image callback; ``None`` lets that callback choose one.
        epochs (int): Positive fit epoch count; defaults to 20.
        use_valset (bool): Build and pass validation data for the selected
            dataset; loaders without a split fall back to test rows.
        show_images (bool): Display callback sample grids during training.
        save_gifs (bool): Save callback denoising animations during training.
            Trajectory callbacks are skipped for no-EMA and VAE/swap models.
        results_path (str | None): Base artifact directory passed to the image
            callback; ``None`` delegates path selection to that callback.
        save_weights (bool): Save final wrapper weights and record their path.
            Dynamic diffusion weights require a paired updated config file;
            training writes it even if ordinary config saving was disabled.
        task (str): ``legacy``, ``generation``, ``joint``, ``classification``,
            or ``continual``.
        seed (int | None): TensorFlow/Keras and dataset split seed.
        verbose (int): Keras and project reporting verbosity.
        patience (int): Early-stopping patience; ``0`` disables it.
        monitor (str | None): Explicit early-stopping metric, or ``None`` for
            the training API default.
        monitor_mode (str): ``"auto"``, ``"min"``, or ``"max"``.
        tensorboard (bool): Write TensorBoard summaries when true.
        tensorboard_path (str | None): Optional TensorBoard root.
        tensorboard_run_name (str | None): Event-file suffix used by HPO to
            encode every sampled value.
        report_every_epoch (bool): Enable compatible diffusion image callbacks.
    """

    project_tag: str | None = None
    epochs: int = 20
    use_valset: bool = True
    show_images: bool = False
    save_gifs: bool = True
    results_path: str | None = "./results"
    save_weights: bool = True
    task: str = "legacy"
    seed: int | None = None
    verbose: int = 1
    patience: int = 0
    monitor: str | None = None
    monitor_mode: str = "auto"
    tensorboard: bool = False
    tensorboard_path: str | None = None
    tensorboard_run_name: str | None = None
    report_every_epoch: bool = True


@dataclass
class ReportingConfig:
    """Post-training evaluation, plot, sample, GIF, and CSV controls.

    Attributes:
        show_history_plot (bool): Display history figures interactively.
        save_history_plot (bool): Save history figures in the result directory.
        final_images_cfg_scale (float): CFG scale for final generation.
        final_images_steps (int): Reverse-diffusion steps for final samples;
            it must satisfy wrapper sampling bounds.
        show_final_images (bool): Display the final sample grid.
        save_final_images (bool): Save the final sample grid as PNG.
        save_final_gifs (bool): Request sampling trajectories and save a GIF;
            VAE/swap sampling has no trajectories, so only images are reported.
        plot_without_20percent (bool): Also plot history after discarding the
            first ``int(0.2 * epochs)`` epochs.
        run_trainset_eval (bool): Evaluate EMA and raw networks on training data.
        run_valset_eval (bool): Evaluate EMA and raw networks on validation
            data; requires a non-``None`` validation dataset.
        evaluate_ensemble_accuracy (bool): Also evaluate ensemble accuracy for
            ``DiffusionClassifier`` and ``DiffusionClassifierV2`` models.
        ensemble_accuracy_kwargs (dict): Options forwarded to
            ``DiffusionClassifier.evaluate_ensemble_accuracy``. Reporting
            selects the raw or EMA network for each evaluation.
        save_csv (bool): Save epoch metrics and enabled evaluations as CSV.
    """

    show_history_plot: bool = False
    save_history_plot: bool = True
    final_images_cfg_scale: float = 3.0
    final_images_steps: int = 1_000
    show_final_images: bool = False
    save_final_images: bool = True
    save_final_gifs: bool = True
    plot_without_20percent: bool = True
    run_trainset_eval: bool = True
    run_valset_eval: bool = True
    evaluate_ensemble_accuracy: bool = False
    ensemble_accuracy_kwargs: dict[str, object] = field(default_factory=dict)
    save_csv: bool = True


@dataclass
class Config:
    """Root configuration object consumed by the training entry point.

    Attributes:
        dataset (DatasetConfig): Input batching and shuffling settings.
        model (ModelConfig): Network/wrapper selection and configuration.
        optimizer (OptimizerConfig): Adam cosine-decay settings.
        training (TrainingConfig): Fit and artifact settings.
        continually_learn (ContinuallyLearnConfig): Class-incremental loop and
            replay settings.
        reporting (ReportingConfig): Post-training reporting settings.
        hpo (dict): Resolved study/trial metadata, sampled values, and selected
            accuracy feedback signal.

    All fields accept either their declared dataclass instance or a mapping in
    ``Config(...)``/YAML input.  Missing top-level or nested fields use
    dataclass defaults; unknown keys raise ``TypeError`` during construction.
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    continually_learn: ContinuallyLearnConfig = field(
        default_factory=ContinuallyLearnConfig
    )
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    hpo: dict = field(default_factory=dict)

    def __post_init__(self: Config) -> None:
        """Convert every supplied top-level mapping to its section dataclass.

        Returns:
            None: Section attributes are normalized in place.
        """

        self.dataset = _section(self.dataset, DatasetConfig)
        self.model = _section(self.model, ModelConfig)
        self.optimizer = _section(self.optimizer, OptimizerConfig)
        self.training = _section(self.training, TrainingConfig)
        self.continually_learn = _section(
            self.continually_learn, 
            ContinuallyLearnConfig
        )
        self.reporting = _section(self.reporting, ReportingConfig)


def _section(
    value: SectionT | Mapping[str, object], 
    section_type: type[SectionT]
) -> SectionT:
    """Return an existing config section or construct one from a mapping.

    Args:
        value (object | Mapping[str, object]): Existing ``section_type``
            instance or keyword mapping.  ``None`` is not treated as an omitted
            section; it fails when expanded.
        section_type (type): Dataclass class to recognize/construct.

    Returns:
        object: ``value`` unchanged when already of ``section_type``;
        otherwise ``section_type(**value)``.

    Raises:
        TypeError: If ``value`` is not a compatible mapping or has unknown keys.
    """

    # Reuse an already constructed typed section unchanged.
    if isinstance(value, section_type):
        return value

    return section_type(**value)


def load_config(
    path: str | os.PathLike[str] | None = None
) -> Config:
    """Load a complete config tree from YAML or dataclass defaults.

    Args:
        path (str | os.PathLike | None): YAML file path.  ``None`` returns
            ``Config()`` and does not read ``configs/default.yaml``.  A YAML
            file may omit sections/fields to receive dataclass defaults, but
            its root must be a mapping.

    Returns:
        Config: Typed root with all nested mappings converted to section
        dataclasses.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        yaml.YAMLError: If YAML syntax is invalid.
        TypeError: If the YAML root/section is not a mapping or contains an
            unknown dataclass field.
    """

    # Return pure dataclass defaults when no YAML path is supplied.
    if path is None:
        return Config()

    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    
    # Require a mapping at the YAML document root.
    if not isinstance(data, Mapping):
        raise TypeError("The YAML document root must be a mapping.")

    return Config(**data)


def save_config(
    config: Config, 
    config_path: str | os.PathLike[str]
) -> None:
    """Serialize a config dataclass tree to sorted YAML.

    Args:
        config (Config): Dataclass root to convert recursively with ``asdict``.
        config_path (str | os.PathLike): Destination file.  Existing content is
            overwritten; parent directories must already exist.

    Returns:
        None.

    Raises:
        TypeError: If ``config`` is not a dataclass instance or contains values
            PyYAML cannot represent.
        OSError: If the destination cannot be opened or written.
    """

    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(asdict(config), file, sort_keys=True)


def run_self_tests() -> dict[str, str]:
    """Run exhaustive, file-local tests for every configuration class.

    The suite checks defaults and complementary overrides, inheritance,
    recursive ``kwargs`` copying, mapping-to-dataclass normalization, distinct
    default factories, YAML full and partial round trips, and representative
    invalid mappings/files.  Temporary files are removed automatically.

    Args:
        None.

    Returns:
        dict[str, str]: One ``"passed"`` entry for each of the twenty classes
        defined by this module.
    """

    from dataclasses import make_dataclass
    from pathlib import Path
    from tempfile import TemporaryDirectory


    mixin_probe_type = make_dataclass(
        "KwargsMixinProbe", 
        [("values", list[int]), ("options", dict[str, object])], 
        bases=(KwargsMixin,)
    )
    original_values = [1, 2]
    mixin_probe = mixin_probe_type(original_values, {"nested": [3]})
    kwargs_copy = mixin_probe.kwargs()
    assert kwargs_copy == {"values": [1, 2], "options": {"nested": [3]}}
    kwargs_copy["values"].append(9)
    kwargs_copy["options"]["nested"].append(4)
    assert original_values == [1, 2]
    assert mixin_probe.options == {"nested": [3]}
    try:
        KwargsMixin().kwargs()
    except TypeError:
        pass
    else:
        raise AssertionError("KwargsMixin.kwargs must reject non-dataclass instances.")

    transformer_defaults = DiffusionTransformerConfig()
    assert transformer_defaults.num_classes == 10
    assert transformer_defaults.timesteps == 1_000
    assert transformer_defaults.use_cfg is True
    transformer_custom = DiffusionTransformerConfig(
        num_classes=0, 
        timesteps=1, 
        use_cfg=False, 
        image_size=8, 
        channels=3, 
        patch_size=4, 
        dim=16, 
        patchify_with_cnn=True, 
        patches_pos_merger_type="concat", 
        patches_conds_merger_type="concat", 
        time_freq_dim=32, 
        time_mlp_ratio=2, 
        mha_key_dim=4, 
        mha_num_heads=2, 
        depth=0, 
        vit_block_mlp_ratio=1, 
        ln_mlp_ratio=3, 
        cls_token_regularizer_kwargs={
            "start": 0,
            "end": 1,
            "mlp_ratio": 2.0,
            "activation_function": "relu",
        },
        use_refiner_cnn=True, 
        refiner_cnn_hidden_dim=7, 
        refiner_cnn_residual=False, 
    )
    assert transformer_custom.kwargs() == asdict(transformer_custom)
    assert transformer_custom.depth == 0 and transformer_custom.use_cfg is False
    assert (
        transformer_custom.cls_token_regularizer_kwargs["activation_function"]
        == "relu"
    )
    transformer_none_values = DiffusionTransformerConfig(
        time_freq_dim=None, 
        time_mlp_ratio=None, 
        mha_key_dim=None, 
        ln_mlp_ratio=None, 
        refiner_cnn_hidden_dim=None, 
    )
    assert transformer_none_values.mha_key_dim is None

    decoder_defaults = DiTDecoderConfig()
    assert decoder_defaults.encoder_output_grid_size is None
    assert decoder_defaults.shift_inputs is True
    assert decoder_defaults.use_decoder_ids == [None]
    decoder_custom = DiTDecoderConfig(
        encoder_output_grid_size=4,
        encoder_output_dim=8,
        shift_inputs=False,
        use_causal_mask=False,
    )
    assert decoder_custom.kwargs()["encoder_output_dim"] == 8

    encoder_decoder_defaults = DiTEncoderDecoderConfig()
    encoder_decoder_custom = DiTEncoderDecoderConfig(
        encoder_kwargs={"depth": 1},
        decoder_kwargs={"depth": 1},
    )
    assert encoder_decoder_defaults.encoder_kwargs is None
    assert encoder_decoder_custom.kwargs()["decoder_kwargs"] == {"depth": 1}

    classifier_defaults = DiTClassifierConfig()
    assert isinstance(classifier_defaults, DiffusionTransformerConfig)
    assert classifier_defaults.aggregate_from_noises is False
    assert classifier_defaults.classifier_only_cls_token is True
    classifier_custom = DiTClassifierConfig(
        num_classes=3, 
        aggregate_from_noises=True, 
        classifier_only_cls_token=False, 
        cls_token_pos_merger_type="concat", 
        clf_cls_token_regularizer_kwargs={
            "start": 0,
            "end": 1,
            "mlp_ratio": 1.5,
            "activation_function": "tanh",
        },
        dropout_rate=0.5, 
    )
    assert classifier_custom.kwargs()["aggregate_from_noises"] is True
    assert classifier_custom.kwargs()["num_classes"] == 3
    assert (
        classifier_custom.kwargs()["clf_cls_token_regularizer_kwargs"][
            "mlp_ratio"
        ] == 1.5
    )

    encoder_decoder_classifier_defaults = DiTEncoderDecoderClassifierConfig()
    assert isinstance(encoder_decoder_classifier_defaults, DiTClassifierConfig)
    encoder_decoder_classifier_custom = DiTEncoderDecoderClassifierConfig(
        encoder_kwargs={"clf_depth": 2},
        decoder_kwargs={"depth": 1},
    )
    assert encoder_decoder_classifier_custom.encoder_kwargs == {"clf_depth": 2}

    unet_defaults = UNetConfig()
    unet_custom = UNetConfig(
        image_size=8,
        widths=[4, 8],
        use_skip_connections=False,
    )
    assert unet_defaults.widths == [32, 64, 96]
    assert unet_custom.kwargs()["widths"] == [4, 8]
    unet_defaults.widths.append(128)
    assert UNetConfig().widths == [32, 64, 96]

    unet_classifier_defaults = UNetClassifierConfig()
    assert isinstance(unet_classifier_defaults, UNetConfig)
    assert unet_classifier_defaults.feature_aggregation_ids_dict == {1: [-1]}
    unet_classifier_custom = UNetClassifierConfig(
        aggregate_from_noises=True,
        clf_depth=2,
        force_global_avg_pooling=False,
    )
    assert unet_classifier_custom.kwargs()["clf_depth"] == 2

    diffusion_defaults = DiffusionModelConfig()
    assert diffusion_defaults.test_network_name == "ema"
    assert diffusion_defaults.swap_noise_image is False
    assert diffusion_defaults.seen_classes == {}
    diffusion_custom = DiffusionModelConfig(
        test_network_name="raw", 
        ema_decay=0.0, 
        scheduler_name="linear", 
        modify_first_t=True, 
        p_uncond=0.0, 
        test_steps=2, 
        test_cfg_scale=1.0, 
        swap_noise_image=True, 
        seen_classes={4: 0},
    )
    assert diffusion_custom.kwargs() == asdict(diffusion_custom)
    assert diffusion_custom.modify_first_t and diffusion_custom.swap_noise_image
    diffusion_defaults.seen_classes[9] = 0
    assert DiffusionModelConfig().seen_classes == {}

    diffusion_classifier_defaults = DiffusionClassifierConfig()
    assert isinstance(diffusion_classifier_defaults, DiffusionModelConfig)
    assert diffusion_classifier_defaults.mask_by_nulls is True
    diffusion_classifier_custom = DiffusionClassifierConfig(
        mask_by_nulls=False, 
        mask_by_t_threshold=True, 
        mask_t_percentage=0, 
        clf_loss_coef=0.0, 
        test_network_name="raw", 
    )
    assert diffusion_classifier_custom.kwargs()["mask_by_t_threshold"] is True
    assert diffusion_classifier_custom.kwargs()["clf_loss_coef"] == 0.0

    diffusion_classifier_v2_defaults = DiffusionClassifierV2Config()
    assert isinstance(diffusion_classifier_v2_defaults, DiffusionClassifierConfig)
    assert diffusion_classifier_v2_defaults.clf_loss_coef == 1.0
    diffusion_classifier_v2_custom = DiffusionClassifierV2Config(
        clf_vars_embedding_ids=[0, 2],
        clf_vars_noise_part_ids=[-1],
        clf_train_noisified_max_timesteps=3,
    )
    assert diffusion_classifier_v2_custom.kwargs()["clf_vars_noise_part_ids"] == [-1]

    vae_defaults = VariationalAutoencoderConfig()
    vae_custom = VariationalAutoencoderConfig(
        data_dim=4,
        hiddens_dims=[],
        conditioned=True,
        class_num=2,
        compile=False,
    )
    assert vae_defaults.hiddens_dims == [16]
    assert vae_custom.kwargs()["class_num"] == 2

    vae_classifier_defaults = VAEClassifierConfig()
    vae_classifier_custom = VAEClassifierConfig(
        data_dim=4,
        hiddens_dims=[],
        alpha=0.0,
    )
    assert "conditioned" not in vae_classifier_defaults.kwargs()
    assert "class_num" not in vae_classifier_defaults.kwargs()
    assert "compile" not in vae_classifier_defaults.kwargs()
    assert vae_classifier_custom.kwargs()["alpha"] == 0.0

    dataset_defaults = DatasetConfig()
    dataset_boundary = DatasetConfig(batch_size=1, shuffle_buffer=0)
    assert dataset_defaults == DatasetConfig(batch_size=128, shuffle_buffer=10_000)
    assert dataset_boundary.batch_size == 1 and dataset_boundary.shuffle_buffer == 0

    transformer_instance = DiffusionTransformerConfig(dim=5)
    model_from_instances = ModelConfig(
        with_classifier=False, 
        show_network_summary=False, 
        weights_path="weights.h5", 
        loss_function="mae", 
        diffusion_transformer=transformer_instance, 
        dit_classifier=classifier_custom, 
        dit_decoder=decoder_custom,
        dit_encoder_decoder=encoder_decoder_custom,
        dit_encoder_decoder_classifier=encoder_decoder_classifier_custom,
        unet=unet_custom,
        unet_classifier=unet_classifier_custom,
        variational_autoencoder=vae_custom,
        vae_classifier=vae_classifier_custom,
        diffusion_model=diffusion_custom, 
        diffusion_classifier=diffusion_classifier_custom, 
        diffusion_classifier_v2=diffusion_classifier_v2_custom,
    )
    assert model_from_instances.diffusion_transformer is transformer_instance
    assert model_from_instances.dit_classifier is classifier_custom
    assert model_from_instances.loss_function == "mae"
    model_from_mappings = ModelConfig(
        diffusion_transformer={"dim": 11, "use_cfg": False}, 
        dit_classifier={"dropout_rate": 0.25}, 
        dit_decoder={"encoder_output_grid_size": 7},
        dit_encoder_decoder={"decoder_kwargs": {"depth": 1}},
        dit_encoder_decoder_classifier={"encoder_kwargs": {"clf_depth": 2}},
        unet={"widths": [4, 8]},
        unet_classifier={"clf_depth": 2},
        variational_autoencoder={"latent_dim": 3},
        vae_classifier={"alpha": 0.5},
        diffusion_model={"test_network_name": "raw"}, 
        diffusion_classifier={"mask_by_nulls": False}, 
        diffusion_classifier_v2={"clf_vars_embedding_ids": [0]},
    )
    assert isinstance(model_from_mappings.diffusion_transformer, DiffusionTransformerConfig)
    assert isinstance(model_from_mappings.dit_classifier, DiTClassifierConfig)
    assert isinstance(model_from_mappings.dit_decoder, DiTDecoderConfig)
    assert isinstance(model_from_mappings.dit_encoder_decoder, DiTEncoderDecoderConfig)
    assert isinstance(
        model_from_mappings.dit_encoder_decoder_classifier,
        DiTEncoderDecoderClassifierConfig,
    )
    assert isinstance(model_from_mappings.unet, UNetConfig)
    assert isinstance(model_from_mappings.unet_classifier, UNetClassifierConfig)
    assert isinstance(
        model_from_mappings.variational_autoencoder,
        VariationalAutoencoderConfig,
    )
    assert isinstance(model_from_mappings.vae_classifier, VAEClassifierConfig)
    assert isinstance(model_from_mappings.diffusion_model, DiffusionModelConfig)
    assert isinstance(model_from_mappings.diffusion_classifier, DiffusionClassifierConfig)
    assert isinstance(
        model_from_mappings.diffusion_classifier_v2,
        DiffusionClassifierV2Config,
    )
    assert model_from_mappings.diffusion_transformer.dim == 11
    assert ModelConfig().diffusion_transformer is not ModelConfig().diffusion_transformer
    try:
        ModelConfig(diffusion_transformer={"unknown": 1})
    except TypeError:
        pass
    else:
        raise AssertionError("Unknown nested model fields must be rejected.")

    optimizer_defaults = OptimizerConfig()
    optimizer_custom = OptimizerConfig(initial_learning_rate=0.0, decay_steps=1)
    assert optimizer_defaults.decay_steps is None
    assert optimizer_custom == OptimizerConfig(0.0, 1)

    continually_learn_defaults = ContinuallyLearnConfig()
    continually_learn_custom = ContinuallyLearnConfig(
        class_num=3,
        remove_prev_classes=False,
        keep_same_model=False,
        use_loaded_opt=False,
        use_buffer=True,
        buffer_kwargs={"maxlen": 8, "sample_num": 2, "insert_num": 2},
        plot_results=False,
        generative_model_kwargs={"train_num": -1, "samples_per_class": 2},
        evaluate_ensemble_accuracy=True,
        ensemble_accuracy_kwargs={"weighted": True, "max_t": 2},
        return_details=True,
    )
    assert continually_learn_defaults.class_num is None
    assert continually_learn_defaults.buffer_kwargs["maxlen"] == 10_000
    assert continually_learn_custom.class_num == 3
    assert continually_learn_custom.use_buffer is True
    assert continually_learn_defaults.evaluate_ensemble_accuracy is False
    assert continually_learn_custom.ensemble_accuracy_kwargs["max_t"] == 2
    continually_learn_defaults.buffer_kwargs["maxlen"] = 1
    continually_learn_defaults.ensemble_accuracy_kwargs["max_t"] = 1
    assert ContinuallyLearnConfig().buffer_kwargs["maxlen"] == 10_000
    assert ContinuallyLearnConfig().ensemble_accuracy_kwargs == {}

    training_defaults = TrainingConfig()
    training_custom = TrainingConfig(
        project_tag="self-test", 
        epochs=1, 
        use_valset=False, 
        show_images=True, 
        save_gifs=False, 
        results_path=None, 
        save_weights=False, 
    )
    assert training_defaults.use_valset is True and training_defaults.save_gifs is True
    assert training_custom.show_images is True and training_custom.results_path is None

    reporting_defaults = ReportingConfig()
    reporting_custom = ReportingConfig(
        show_history_plot=True, 
        save_history_plot=False, 
        final_images_cfg_scale=0.0, 
        final_images_steps=2, 
        show_final_images=True, 
        save_final_images=False, 
        save_final_gifs=False, 
        plot_without_20percent=False, 
        run_trainset_eval=False, 
        run_valset_eval=False, 
        evaluate_ensemble_accuracy=True,
        ensemble_accuracy_kwargs={"weighted": True, "max_t": 2},
        save_csv=False,
    )
    assert reporting_defaults.save_history_plot is True
    assert reporting_custom.show_history_plot is True
    assert reporting_defaults.evaluate_ensemble_accuracy is False
    assert reporting_custom.ensemble_accuracy_kwargs["max_t"] == 2
    reporting_defaults.ensemble_accuracy_kwargs["max_t"] = 1
    assert ReportingConfig().ensemble_accuracy_kwargs == {}
    assert not any((
        reporting_custom.save_history_plot, 
        reporting_custom.save_final_images, 
        reporting_custom.save_final_gifs, 
        reporting_custom.plot_without_20percent, 
        reporting_custom.run_trainset_eval, 
        reporting_custom.run_valset_eval, 
        reporting_custom.save_csv,
    ))

    default_config_a = Config()
    default_config_b = Config()
    assert isinstance(default_config_a.dataset, DatasetConfig)
    assert isinstance(default_config_a.model, ModelConfig)
    assert default_config_a.model.loss_function == "mse"
    assert isinstance(default_config_a.optimizer, OptimizerConfig)
    assert isinstance(default_config_a.training, TrainingConfig)
    assert isinstance(
        default_config_a.continually_learn, ContinuallyLearnConfig
    )
    assert isinstance(default_config_a.reporting, ReportingConfig)
    assert default_config_a.dataset is not default_config_b.dataset
    assert default_config_a.model is not default_config_b.model
    assert default_config_a.continually_learn is not \
        default_config_b.continually_learn
    mapped_config = Config(
        dataset={"batch_size": 2, "shuffle_buffer": 0}, 
        model={
            "with_classifier": False, 
            "loss_function": "mae", 
            "diffusion_transformer": {"depth": 0}
        }, 
        optimizer={"initial_learning_rate": 1e-4, "decay_steps": 3}, 
        training={"epochs": 1, "save_weights": False}, 
        continually_learn={
            "class_num": 4,
            "plot_results": False,
            "evaluate_ensemble_accuracy": True
        },
        reporting={
            "run_trainset_eval": False,
            "run_valset_eval": False,
            "ensemble_accuracy_kwargs": {"max_t": 4}
        },
    )
    assert mapped_config.dataset.batch_size == 2
    assert mapped_config.model.loss_function == "mae"
    assert mapped_config.model.diffusion_transformer.depth == 0
    assert mapped_config.optimizer.decay_steps == 3
    assert mapped_config.training.epochs == 1
    assert mapped_config.continually_learn.class_num == 4
    assert mapped_config.continually_learn.evaluate_ensemble_accuracy is True
    assert mapped_config.reporting.run_trainset_eval is False
    assert mapped_config.reporting.ensemble_accuracy_kwargs["max_t"] == 4
    assert _section(mapped_config.dataset, DatasetConfig) is mapped_config.dataset
    assert _section({"batch_size": 4}, DatasetConfig).batch_size == 4
    for invalid_value in (None, 1, "dataset"):
        try:
            _section(invalid_value, DatasetConfig)
        except TypeError:
            pass
        else:
            raise AssertionError("Non-mapping configuration sections must fail.")

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        full_path = temp_path / "full.yaml"
        save_config(mapped_config, full_path)
        full_text = full_path.read_text(encoding="utf-8")
        assert full_text.index("dataset:") < full_text.index("training:")
        loaded_full = load_config(full_path)
        assert loaded_full == mapped_config

        partial_path = temp_path / "partial.yaml"
        partial_path.write_text(
            "dataset:\n  batch_size: 3\nmodel:\n  diffusion_model:\n"
            "    scheduler_name: linear\n",
            encoding="utf-8",
        )
        loaded_partial = load_config(partial_path)
        assert loaded_partial.dataset.batch_size == 3
        assert loaded_partial.dataset.shuffle_buffer == 10_000
        assert loaded_partial.model.diffusion_model.scheduler_name == "linear"

        invalid_yaml_path = temp_path / "invalid.yaml"
        invalid_yaml_path.write_text("dataset: [", encoding="utf-8")
        try:
            load_config(invalid_yaml_path)
        except yaml.YAMLError:
            pass
        else:
            raise AssertionError("Malformed YAML must raise yaml.YAMLError.")

        non_mapping_path = temp_path / "list.yaml"
        non_mapping_path.write_text("- one\n- two\n", encoding="utf-8")
        try:
            load_config(non_mapping_path)
        except TypeError:
            pass
        else:
            raise AssertionError("A non-mapping YAML root must be rejected.")

        unknown_path = temp_path / "unknown.yaml"
        unknown_path.write_text("unknown: true\n", encoding="utf-8")
        try:
            load_config(unknown_path)
        except TypeError:
            pass
        else:
            raise AssertionError("Unknown top-level YAML keys must be rejected.")

        try:
            load_config(temp_path / "missing.yaml")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing configuration files must fail.")

        try:
            save_config({"not": "a dataclass"}, temp_path / "invalid-save.yaml")
        except TypeError:
            pass
        else:
            raise AssertionError("save_config must reject non-dataclass roots.")

    assert load_config() == Config()

    return {
        "KwargsMixin": "passed", 
        "DiffusionTransformerConfig": "passed", 
        "DiTDecoderConfig": "passed",
        "DiTEncoderDecoderConfig": "passed",
        "DiTClassifierConfig": "passed", 
        "DiTEncoderDecoderClassifierConfig": "passed",
        "UNetConfig": "passed",
        "UNetClassifierConfig": "passed",
        "DiffusionModelConfig": "passed", 
        "DiffusionClassifierConfig": "passed", 
        "DiffusionClassifierV2Config": "passed",
        "VariationalAutoencoderConfig": "passed",
        "VAEClassifierConfig": "passed",
        "DatasetConfig": "passed", 
        "ModelConfig": "passed", 
        "OptimizerConfig": "passed", 
        "ContinuallyLearnConfig": "passed",
        "TrainingConfig": "passed", 
        "ReportingConfig": "passed", 
        "Config": "passed", 
    }


# Run this module's executable self-test entry point when invoked directly.
if __name__ == "__main__":
    print(run_self_tests())
