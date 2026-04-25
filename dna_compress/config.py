from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    implementation: str = "megabyte"
    variant: str = "dna_gpt0.1b_m"
    pretrained_weight_path: str | None = None
    vocab_size: int = 259
    patch_size: int = 8
    global_dim: int = 64
    local_dim: int = 128
    seq_length: int = 512
    global_heads: int = 8
    global_layers: int = 4
    local_heads: int = 4
    local_layers: int = 2
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    input_causal_conv_kernel_size: int = 1
    initializer_range: float = 0.02
    pad_id: int = 257
    eos_id: int = 258
    nugget_backbone: str = "bart"
    nugget_bart_config_path: str = "configs/hf/facebook_bart_base_config.json"
    nugget_bart_d_model: int | None = None
    nugget_bart_encoder_layers: int | None = None
    nugget_bart_decoder_layers: int | None = None
    nugget_bart_encoder_attention_heads: int | None = None
    nugget_bart_decoder_attention_heads: int | None = None
    nugget_bart_encoder_ffn_dim: int | None = None
    nugget_bart_decoder_ffn_dim: int | None = None
    nugget_bart_dropout: float | None = None
    nugget_bart_attention_dropout: float | None = None
    nugget_bart_activation_dropout: float | None = None
    nugget_bart_max_position_embeddings: int | None = None
    nugget_t5_d_model: int = 512
    nugget_t5_d_ff: int = 2048
    nugget_t5_num_layers: int = 6
    nugget_t5_num_decoder_layers: int = 6
    nugget_t5_num_heads: int = 8
    nugget_t5_d_kv: int | None = None
    nugget_t5_dropout_rate: float = 0.1
    nugget_ratio: float = 0.25
    nugget_scorer_layer: int = 3
    nugget_residual_start: int = 0
    nugget_residual_end: int = -1
    nugget_value_ffn: bool = True
    nugget_straight_through: bool = True
    nugget_hidden_mode: str = "runtime_hidden"
    nugget_hidden_storage_dtype: str = "runtime"


@dataclass
class DataConfig:
    dataset_dir: str = "datasets/DNACorpus"
    species: list[str] = field(default_factory=lambda: ["HoSa"])
    species_prefix_map: dict[str, str] = field(default_factory=dict)
    nugget_tokenizer: str = "byte"
    sequence_source_mode: str = "auto"
    multi_sequence_mode: str = "separate"
    sequence_include_map: dict[str, list[str]] = field(default_factory=dict)
    clean_cache_enabled: bool = True
    clean_cache_dir: str | None = None
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    max_train_bytes_per_species: int | None = None
    max_val_bytes_per_species: int | None = None
    max_test_bytes_per_species: int | None = None
    train_samples_per_epoch: int = 2_048
    train_sampling_strategy: str = "proportional"
    token_merge_size: int = 1
    token_merge_alphabet: str = "ACGTN"
    compression_sample_bytes: int = 16_384


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"
    dtype: str = "float16"
    init_from: str = "scratch"
    epochs: int = 1
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    lr_scheduler: str = "none"
    lr_warmup_steps: int = 0
    lr_min_ratio: float = 0.0
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    log_interval: int = 25
    eval_interval: int = 100
    gpu_ids: list[int] | None = None


@dataclass
class OutputConfig:
    run_name: str = "dna_megabyte_quick"
    output_dir: str = "outputs/dna_megabyte_quick"
    wandb_enabled: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_name: str | None = None
    wandb_group: str = ""
    wandb_tags: list[str] = field(default_factory=list)
    wandb_mode: str = "online"


@dataclass
class ArithmeticCodingConfig:
    frequency_total: int | None = None
    target_uniform_mass: float = 0.01
    coding_mode: str = "model_symbol"
    merge_size: int = 1


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    arithmetic: ArithmeticCodingConfig = field(default_factory=ArithmeticCodingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(cls, values: dict[str, Any] | None):
    if values is None:
        return cls()
    merged = cls()
    for key, value in values.items():
        setattr(merged, key, value)
    return merged


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        train=_merge_dataclass(TrainConfig, raw.get("train")),
        output=_merge_dataclass(OutputConfig, raw.get("output")),
        arithmetic=_merge_dataclass(ArithmeticCodingConfig, raw.get("arithmetic")),
    )


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
