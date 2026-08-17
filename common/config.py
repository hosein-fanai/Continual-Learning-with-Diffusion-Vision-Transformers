"""Dataclass configuration tree and YAML serialization for MNIST training.

Leaf model sections expose :meth:`KwargsMixin.kwargs`; ``common.train`` passes
those mappings directly to transformer and wrapper constructors without key
translation or validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


class KwargsMixin:
    """Convert a configuration dataclass into constructor keyword arguments."""

    def kwargs(self) -> dict[str, Any]:
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

    num_classes: int = 10
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
    vit_block_ids: list[int | None] = field(default_factory=lambda: [None])
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
        default_factory=lambda: {"start": 0, "end": 1}
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
        default_factory=lambda: {1: [-1]}
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
        default_factory=lambda: {-1: [-1]}
    )
    clf_connection_kwargs: dict | None = None
    clf_cross_attention_ids_dict: dict = field(default_factory=dict)
    clf_cross_attention_kwargs: dict | None = None
    clf_cross_attention_plug_type: str | None = None
    clf_vit_block_ids: list[int | None] = field(default_factory=lambda: [None])
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
class DiffusionModelConfig(KwargsMixin):
    """Arguments forwarded to ``DiffusionModel`` except ``network``."""

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
class DatasetConfig:
    """MNIST input-pipeline settings.

    Attributes:
        batch_size (int): Positive examples per batch; defaults to 128.
        shuffle_buffer (int): Training shuffle-buffer capacity.  Values above
            zero enable shuffling; ``0`` disables it.  Defaults to 10,000.
        trainset_len (int): Number of training batches, added dynamically by
            :func:`common.train.get_datasets`; it is not a constructor field.
    """

    batch_size: int = 128
    shuffle_buffer: int = 10_000


@dataclass
class ModelConfig:
    """Select the raw network, wrapper, visibility, and initial weights.

    Attributes:
        with_classifier (bool): Build ``DiTClassifier`` inside
            ``DiffusionClassifier`` when true; otherwise build
            ``DiffusionTransformer`` inside ``DiffusionModel``.
        show_network_summary (bool): Print the wrapper/network summary after
            construction.
        weights_path (str | None): Keras weights file loaded after construction,
            or ``None`` for fresh weights.  Training updates this field to its
            saved ``model.weights.h5`` path.
        diffusion_transformer (DiffusionTransformerConfig): Raw denoising
            network settings used only when ``with_classifier=False``.
        dit_classifier (DiTClassifierConfig): Raw joint network settings used
            only when ``with_classifier=True``.
        diffusion_model (DiffusionModelConfig): Process-wrapper settings used
            only when ``with_classifier=False``.
        diffusion_classifier (DiffusionClassifierConfig): Classifier wrapper
            settings used only when ``with_classifier=True``.
    """

    with_classifier: bool = True
    show_network_summary: bool = True
    weights_path: str | None = None

    diffusion_transformer: DiffusionTransformerConfig = field(
        default_factory=DiffusionTransformerConfig
    )
    dit_classifier: DiTClassifierConfig = field(
        default_factory=DiTClassifierConfig
    )
    diffusion_model: DiffusionModelConfig = field(
        default_factory=DiffusionModelConfig
    )
    diffusion_classifier: DiffusionClassifierConfig = field(
        default_factory=DiffusionClassifierConfig
    )

    def __post_init__(self):
        """Convert nested model-section mappings to typed dataclass instances.

        Returns:
            None: The four nested attributes are normalized in place.  Existing
            instances are preserved; mappings are expanded as constructor
            keywords and may omit fields to use their defaults.
        """

        self.diffusion_transformer = _section(self.diffusion_transformer, DiffusionTransformerConfig)
        self.dit_classifier = _section(self.dit_classifier, DiTClassifierConfig)
        self.diffusion_model = _section(self.diffusion_model, DiffusionModelConfig)
        self.diffusion_classifier = _section(self.diffusion_classifier, DiffusionClassifierConfig)


@dataclass
class OptimizerConfig:
    """Adam learning-rate schedule settings.

    Attributes:
        initial_learning_rate (float): Initial cosine-decay learning rate;
            defaults to ``5e-3``.
        decay_steps (int | None): Positive cosine-decay duration in optimizer
            steps.  ``None`` is replaced during model construction with
            ``epochs * trainset_len``.
    """

    initial_learning_rate: float = 5e-3
    decay_steps: int | None = None


@dataclass
class TrainingConfig:
    """Fit-loop, callback, and artifact-persistence settings.

    Attributes:
        project_tag (str | None): Optional result-run identifier passed to the
            image callback; ``None`` lets that callback choose one.
        epochs (int): Positive fit epoch count; defaults to 20.
        use_valset (bool): Build/pass MNIST test data as validation data.
        show_images (bool): Display callback sample grids during training.
        save_gifs (bool): Save callback denoising animations during training.
            Trajectory callbacks are skipped for no-EMA and VAE/swap models.
        results_path (str | None): Base artifact directory passed to the image
            callback; ``None`` delegates path selection to that callback.
        save_weights (bool): Save final wrapper weights and record their path.
    """

    project_tag: str | None = None
    epochs: int = 20
    use_valset: bool = True
    show_images: bool = False
    save_gifs: bool = True
    results_path: str | None = "./results"
    save_weights: bool = True


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
        save_history_csv (bool): Save epoch metric series as CSV.
        save_evals_csv (bool): Save enabled evaluation dictionaries as CSV.
        results_path (str): Actual run directory, added dynamically by
            :func:`common.train.train_model`; it is not a constructor field.
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
    save_history_csv: bool = True
    save_evals_csv: bool = True


@dataclass
class Config:
    """Root configuration object consumed by the training entry point.

    Attributes:
        dataset (DatasetConfig): Input batching and shuffling settings.
        model (ModelConfig): Network/wrapper selection and configuration.
        optimizer (OptimizerConfig): Adam cosine-decay settings.
        training (TrainingConfig): Fit and artifact settings.
        reporting (ReportingConfig): Post-training reporting settings.

    All fields accept either their declared dataclass instance or a mapping in
    ``Config(...)``/YAML input.  Missing top-level or nested fields use
    dataclass defaults; unknown keys raise ``TypeError`` during construction.
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def __post_init__(self):
        """Convert every supplied top-level mapping to its section dataclass.

        Returns:
            None: Section attributes are normalized in place.
        """
        self.dataset = _section(self.dataset, DatasetConfig)
        self.model = _section(self.model, ModelConfig)
        self.optimizer = _section(self.optimizer, OptimizerConfig)
        self.training = _section(self.training, TrainingConfig)
        self.reporting = _section(self.reporting, ReportingConfig)


def _section(value, section_type):
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

    if isinstance(value, section_type):
        return value

    return section_type(**value)


def load_config(path: str | None = None) -> Config:
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

    if path is None:
        return Config()

    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    
    return Config(**data)


def save_config(config: Config, config_path: str):
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

    with open(config_path, "w") as file:
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
        dict[str, str]: One ``"passed"`` entry for each of the eleven classes
        defined by this module.
    """

    from dataclasses import make_dataclass
    from pathlib import Path
    from tempfile import TemporaryDirectory

    mixin_probe_type = make_dataclass(
        "KwargsMixinProbe",
        [("values", list[int]), ("options", dict[str, object])],
        bases=(KwargsMixin,),
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
        use_refiner_cnn=True, 
        refiner_cnn_hidden_dim=7, 
        refiner_cnn_residual=False, 
    )
    assert transformer_custom.kwargs() == asdict(transformer_custom)
    assert transformer_custom.depth == 0 and transformer_custom.use_cfg is False
    transformer_none_values = DiffusionTransformerConfig(
        time_freq_dim=None, 
        time_mlp_ratio=None, 
        mha_key_dim=None, 
        ln_mlp_ratio=None, 
        refiner_cnn_hidden_dim=None, 
    )
    assert transformer_none_values.mha_key_dim is None

    classifier_defaults = DiTClassifierConfig()
    assert isinstance(classifier_defaults, DiffusionTransformerConfig)
    assert classifier_defaults.aggregate_from_noises is False
    assert classifier_defaults.classifier_only_cls_token is True
    classifier_custom = DiTClassifierConfig(
        num_classes=3, 
        aggregate_from_noises=True, 
        classifier_only_cls_token=False, 
        cls_token_pos_merger_type="concat", 
        dropout_rate=0.5, 
    )
    assert classifier_custom.kwargs()["aggregate_from_noises"] is True
    assert classifier_custom.kwargs()["num_classes"] == 3

    diffusion_defaults = DiffusionModelConfig()
    assert diffusion_defaults.test_network_name == "ema"
    assert diffusion_defaults.swap_noise_image is False
    diffusion_custom = DiffusionModelConfig(
        test_network_name="raw", 
        ema_decay=0.0, 
        scheduler_name="linear", 
        modify_first_t=True, 
        p_uncond=0.0, 
        test_steps=2, 
        test_cfg_scale=1.0, 
        swap_noise_image=True, 
    )
    assert diffusion_custom.kwargs() == asdict(diffusion_custom)
    assert diffusion_custom.modify_first_t and diffusion_custom.swap_noise_image

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

    dataset_defaults = DatasetConfig()
    dataset_boundary = DatasetConfig(batch_size=1, shuffle_buffer=0)
    assert dataset_defaults == DatasetConfig(batch_size=128, shuffle_buffer=10_000)
    assert dataset_boundary.batch_size == 1 and dataset_boundary.shuffle_buffer == 0

    transformer_instance = DiffusionTransformerConfig(dim=5)
    model_from_instances = ModelConfig(
        with_classifier=False, 
        show_network_summary=False, 
        weights_path="weights.h5", 
        diffusion_transformer=transformer_instance, 
        dit_classifier=classifier_custom, 
        diffusion_model=diffusion_custom, 
        diffusion_classifier=diffusion_classifier_custom, 
    )
    assert model_from_instances.diffusion_transformer is transformer_instance
    assert model_from_instances.dit_classifier is classifier_custom
    model_from_mappings = ModelConfig(
        diffusion_transformer={"dim": 11, "use_cfg": False}, 
        dit_classifier={"dropout_rate": 0.25}, 
        diffusion_model={"test_network_name": "raw"}, 
        diffusion_classifier={"mask_by_nulls": False}, 
    )
    assert isinstance(model_from_mappings.diffusion_transformer, DiffusionTransformerConfig)
    assert isinstance(model_from_mappings.dit_classifier, DiTClassifierConfig)
    assert isinstance(model_from_mappings.diffusion_model, DiffusionModelConfig)
    assert isinstance(model_from_mappings.diffusion_classifier, DiffusionClassifierConfig)
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
        save_history_csv=False, 
        save_evals_csv=False, 
    )
    assert reporting_defaults.save_history_plot is True
    assert reporting_custom.show_history_plot is True
    assert not any((
        reporting_custom.save_history_plot, 
        reporting_custom.save_final_images, 
        reporting_custom.save_final_gifs, 
        reporting_custom.plot_without_20percent, 
        reporting_custom.run_trainset_eval, 
        reporting_custom.run_valset_eval, 
        reporting_custom.save_history_csv, 
        reporting_custom.save_evals_csv, 
    ))

    default_config_a = Config()
    default_config_b = Config()
    assert isinstance(default_config_a.dataset, DatasetConfig)
    assert isinstance(default_config_a.model, ModelConfig)
    assert isinstance(default_config_a.optimizer, OptimizerConfig)
    assert isinstance(default_config_a.training, TrainingConfig)
    assert isinstance(default_config_a.reporting, ReportingConfig)
    assert default_config_a.dataset is not default_config_b.dataset
    assert default_config_a.model is not default_config_b.model
    mapped_config = Config(
        dataset={"batch_size": 2, "shuffle_buffer": 0}, 
        model={"with_classifier": False, "diffusion_transformer": {"depth": 0}}, 
        optimizer={"initial_learning_rate": 1e-4, "decay_steps": 3}, 
        training={"epochs": 1, "save_weights": False}, 
        reporting={"run_trainset_eval": False, "run_valset_eval": False}, 
    )
    assert mapped_config.dataset.batch_size == 2
    assert mapped_config.model.diffusion_transformer.depth == 0
    assert mapped_config.optimizer.decay_steps == 3
    assert mapped_config.training.epochs == 1
    assert mapped_config.reporting.run_trainset_eval is False
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
        "DiTClassifierConfig": "passed", 
        "DiffusionModelConfig": "passed", 
        "DiffusionClassifierConfig": "passed", 
        "DatasetConfig": "passed", 
        "ModelConfig": "passed", 
        "OptimizerConfig": "passed", 
        "TrainingConfig": "passed", 
        "ReportingConfig": "passed", 
        "Config": "passed", 
    }


if __name__ == "__main__":
    print(run_self_tests())
