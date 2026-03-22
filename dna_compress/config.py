from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    implementation: str = "megabyte"
    vocab_size: int = 259
    patch_size: int = 8
    global_dim: int = 64
    local_dim: int = 128
    seq_length: int = 512
    global_heads: int = 8
    global_layers: int = 4
    local_heads: int = 4
    local_layers: int = 2
    initializer_range: float = 0.02
    pad_id: int = 257
    eos_id: int = 258


@dataclass
class DataConfig:
    dataset_dir: str = "datasets/DNACorpus"
    species: list[str] = field(default_factory=lambda: ["HoSa"])
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    max_train_bytes_per_species: int | None = 8_388_608
    max_val_bytes_per_species: int | None = 1_048_576
    max_test_bytes_per_species: int | None = 131_072
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
    log_interval: int = 25
    eval_interval: int = 100


@dataclass
class OutputConfig:
    run_name: str = "dna_megabyte_quick"
    output_dir: str = "outputs/dna_megabyte_quick"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

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
    )


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
