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
    """Legacy compact configuration for a diffusion-transformer network.

    Depth 0 is the network stem: patch/time/label embedding and initial token
    merging before repeated processing.  ``depth=N`` requests processing
    depths 1 through N; ``depth=0`` leaves only the stem and output path.

    Attributes:
        num_classes (int): Number of real classes; defaults to 10.  With CFG,
            the network additionally reserves a null-label embedding.
        timesteps (int): Discrete diffusion schedule length; defaults to 1,000.
        use_cfg (bool): Enable classifier-free guidance conditioning and its
            extra null label; defaults to true.
        image_size (int): Native square image side; defaults to 28 and must be
            divisible by ``patch_size`` downstream.
        channels (int): Input and predicted image channels; defaults to 1.
        patch_size (int): Square patch side; defaults to 2.
        dim (int): Base token feature width; defaults to 32.
        project_with_cnn (bool): Legacy name for convolutional patch projection
            (current constructor name: ``patchify_with_cnn``).
        pos_plug_method (str): Legacy position-merging control, normally
            ``"add"`` or ``"concat"`` (current name:
            ``patches_pos_merger_type``).
        conds_plug_method (str): Legacy condition-merging control, normally
            ``"add"`` or ``"concat"`` (current API separates
            ``conds_merger_type`` and ``patches_conds_merger_type``).
        t_emb_freq_dim (int | None): Legacy sinusoidal time-frequency width;
            ``None`` delegates sizing (current name: ``time_freq_dim``).
        t_emb_mlp_ratio (int | None): Legacy time-embedding MLP expansion;
            ``None`` disables/delegates it (current name: ``time_mlp_ratio``).
        mha_key_dim (int | None): Per-head attention key width; ``None`` lets
            the network derive it.
        num_heads (int): Legacy attention-head count (current name:
            ``mha_num_heads``); defaults to 4.
        depth (int): Number of repeated processing depths after depth 0;
            defaults to 3.
        mha_mlp_ratio (int): Legacy transformer-block MLP expansion ratio
            (current name: ``vit_block_mlp_ratio``); defaults to 4.
        ln_mlp_ratio (int | None): Optional adaptive layer-normalization MLP
            expansion; ``None`` delegates/disables according to the layer.
        use_final_cnn (bool): Legacy final-refinement switch (current name:
            ``use_refiner_cnn``).
        final_cnn_hidden_dim (int | None): Legacy refiner hidden channel count
            (current name: ``refiner_cnn_hidden_dim``).
        final_cnn_residual (bool): Legacy residual-refiner switch (current name:
            ``refiner_cnn_residual``).

    Note:
        :meth:`kwargs` emits field names verbatim.  The fields explicitly
        marked ``legacy`` are not accepted parameters of the current
        ``DiffusionTransformer`` constructor and are not automatically mapped;
        use that constructor's documented names for an executable current
        configuration.
    """
    num_classes: int = 10
    timesteps: int = 1_000
    use_cfg: bool = True
    image_size: int = 28
    channels: int = 1
    patch_size: int = 2
    dim: int = 32
    project_with_cnn: bool = False
    pos_plug_method: str = "add"
    conds_plug_method: str = "add"
    t_emb_freq_dim: int = 256
    t_emb_mlp_ratio: int | None = None
    mha_key_dim: int | None = None
    num_heads: int = 4
    depth: int = 3
    mha_mlp_ratio: int = 4
    ln_mlp_ratio: int | None = None
    use_final_cnn: bool = False
    final_cnn_hidden_dim: int | None = None
    final_cnn_residual: bool = True


@dataclass
class DiTClassifierConfig(DiffusionTransformerConfig):
    """Legacy transformer configuration extended with classifier controls.

    Attributes:
        aggregate_from_noises (bool): Aggregate classifier features from
            reconstructed/noise-space inputs rather than only transformer
            features; defaults to false and remains a current constructor key.
        prepend_cls_token (bool): Legacy switch for a learned classifier token;
            the current API expresses this through ``classifier_only_cls_token``
            and ``clf_cls_token_type``.
        cls_token_pos_plug_method (str): Legacy ``"add"``/``"concat"`` class
            token position merger (current name:
            ``cls_token_pos_merger_type``).
        clf_dropout_rate (float): Legacy classifier dropout fraction in
            ``[0, 1)`` (current name: ``dropout_rate``).

    Note:
        It inherits the legacy fields and forwarding caveat from
        :class:`DiffusionTransformerConfig`.  ``kwargs()`` performs no alias
        conversion.
    """
    aggregate_from_noises: bool = False
    prepend_cls_token: bool = True
    cls_token_pos_plug_method: str = "add"
    clf_dropout_rate: float = 0.0


@dataclass
class DiffusionModelConfig(KwargsMixin):
    """Runtime diffusion-process options for the wrapper around a network.

    The ``diffusion.models.transformer`` object defines the neural architecture
    and one denoising/classification forward pass.  The
    ``diffusion.models.wrapper`` object owns schedules, noising, EMA, custom
    train/test steps, and iterative sampling around that network.

    Attributes:
        test_network_name (str): ``"ema"`` evaluates/samples with the
            exponential-moving-average copy; ``"raw"`` uses live weights.
        ema_decay (float): EMA coefficient in ``[0, 1)``; defaults to 0.999.
        scheduler_name (str): Schedule identifier accepted by
            ``diffusion.schedulers.make_schedule``: ``"linear"``,
            ``"scaled_linear"``, ``"squaredcos_cap_v2"``,
            ``"clipped_cosine"``, ``"sigmoid"``, ``"quadratic"``, ``"ve"``,
            ``"karras"``, ``"sub_vp"``, or ``"logistic"``.
        modify_first_t (bool): If true, replace the first schedule point with a
            clean-signal endpoint after schedule creation.
        p_uncond (float): Probability in ``[0, 1]`` of replacing a training
            label with the CFG null label; forced to 0 if CFG is disabled.
        test_steps (int): Default reverse-sampling step count, from 2 through
            ``timesteps``.
        test_cfg_scale (float): Default classifier-free guidance scale.  ``1``
            is unguided conditional prediction; values greater than 1 amplify
            conditioning and CFG-disabled networks force it to 1.
        swap_noise_image (bool): Switch to the wrapper's VAE-oriented path in
            which the noised image tensor also serves as the noise target.
    """
    test_network_name: str = "ema"
    ema_decay: float = 0.999
    scheduler_name: str = "clipped_cosine"
    modify_first_t: bool = False
    p_uncond: float = 0.1
    test_steps: int = 50
    test_cfg_scale: float = 4.0
    swap_noise_image: bool = False


@dataclass
class DiffusionClassifierConfig(DiffusionModelConfig):
    """Legacy diffusion-wrapper configuration with classifier-loss masking.

    Attributes:
        mask_by_nulls (bool): Include classification loss only for CFG-null
            rows when true; requires ``p_uncond > 0``.
        mask_by_t_threshold (bool): Also retain only timesteps at or below the
            percentage-derived threshold when true.
        mask_t_percentage (int): Percentage used as
            ``int(value / 100 * timesteps)``; normally in ``[0, 100]``.
        lambda_ (float): Legacy classifier-loss coefficient (current
            ``DiffusionClassifier`` name: ``clf_loss_coef``), default 0.0086.

    Note:
        ``lambda_`` is emitted unchanged by :meth:`kwargs` and is not accepted
        by the current wrapper constructor; use ``clf_loss_coef`` with the
        current API.
    """
    mask_by_nulls: bool = True
    mask_by_t_threshold: bool = False
    mask_t_percentage: int = 70
    lambda_: float = 8.6e-3


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
        save_final_gifs (bool): Request sampling trajectories and save a GIF.
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
