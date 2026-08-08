from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


class KwargsMixin:
    """Small helper for passing config dataclasses into constructors."""

    def kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiffusionTransformerConfig(KwargsMixin):
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
    prepend_cls_token: bool = True
    cls_token_pos_plug_method: str = "add"
    clf_dropout_rate: float = 0.0


@dataclass
class DiffusionModelConfig(KwargsMixin):
    test_network_name: str = "ema"
    ema_decay: float = 0.999
    scheduler_name: str = "clipped_cosine"
    p_uncond: float = 0.1
    test_steps: int = 50
    test_cfg_scale: float = 4.0


@dataclass
class DiffusionClassifierConfig(DiffusionModelConfig):
    mask_by_nulls: bool = True
    mask_by_t_threshold: bool = False
    mask_t_percentage: int = 70
    lambda_: float = 8.6e-3


@dataclass
class DatasetConfig:
    batch_size: int = 128
    shuffle_buffer: int = 10_000


@dataclass
class ModelConfig:
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
        self.diffusion_transformer = _section(self.diffusion_transformer, DiffusionTransformerConfig)
        self.dit_classifier = _section(self.dit_classifier, DiTClassifierConfig)
        self.diffusion_model = _section(self.diffusion_model, DiffusionModelConfig)
        self.diffusion_classifier = _section(self.diffusion_classifier, DiffusionClassifierConfig)


@dataclass
class OptimizerConfig:
    initial_learning_rate: float = 5e-3
    decay_steps: int | None = None


@dataclass
class TrainingConfig:
    project_tag: str | None = None
    epochs: int = 20
    use_valset: bool = True
    show_images: bool = False
    save_gifs: bool = True
    results_path: str | None = "./results"
    save_weights: bool = True


@dataclass
class ReportingConfig:
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
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def __post_init__(self):
        self.dataset = _section(self.dataset, DatasetConfig)
        self.model = _section(self.model, ModelConfig)
        self.optimizer = _section(self.optimizer, OptimizerConfig)
        self.training = _section(self.training, TrainingConfig)
        self.reporting = _section(self.reporting, ReportingConfig)


def _section(value, section_type):
    if isinstance(value, section_type):
        return value

    return section_type(**value)


def load_config(path: str | None = None) -> Config:
    if path is None:
        return Config()

    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    
    return Config(**data)


def save_config(config: Config, config_path: str):
    with open(config_path, "w") as file:
        yaml.safe_dump(asdict(config), file, sort_keys=True)

