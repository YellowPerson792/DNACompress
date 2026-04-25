from __future__ import annotations

"""Train or evaluate Nugget encoder-decoder models on DNACorpus.

Examples:

  # Train/evaluate a BART Nugget autoencoder.
  python scripts/run_nugget_experiment.py \
    --config configs/dna_nugget_modified.json \
    --mode all \
    --implementation nugget \
    --nugget-backbone bart \
    --nugget-tokenizer fixed_kmer \
    --token-merge-size 6 \
    --token-merge-alphabet ACGTN \
    --nugget-ratio 0.7 \
    --nugget-scorer-layer 3 \
    --nugget-residual-start 0 \
    --nugget-residual-end -1 \
    --seq-length 512 \
    --dataset-dir datasets/DNACorpus \
    --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
    --train-ratio 0.6 \
    --val-ratio 0.2 \
    --test-ratio 0.2 \
    --train-samples-per-epoch 2500000 \
    --compression-sample-bytes 100000 \
    --arithmetic-coding-mode model_symbol \
    --arithmetic-merge-size 1 \
    --device cuda \
    --dtype bfloat16 \
    --init-from scratch \
    --epochs 1 \
    --batch-size 32 \
    --eval-batch-size 32 \
    --learning-rate 5e-5 \
    --weight-decay 0 \
    --lr-scheduler cosine \
    --lr-warmup-steps 0 \
    --lr-min-ratio 0.1 \
    --grad-clip-norm 1.0 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --persistent-workers \
    --pin-memory \
    --log-interval 25 \
    --eval-interval 2000 \
    --print-config \
    --run-name dna_nugget_bart_r0.7 \
    --wandb-project dna-compress \
    --wandb-name dna_nugget_bart

  # Train a T5 Nugget autoencoder.
  python scripts/run_nugget_experiment.py \
    --config configs/dna_nugget_quick.json \
    --mode all \
    --nugget-backbone t5 \
    --nugget-tokenizer fixed_kmer \
    --token-merge-size 3 \
    --token-merge-alphabet ACGTN \
    --arithmetic-coding-mode fixed_token_units \
    --arithmetic-merge-size 1 \
    --seq-length 1024 \
    --species HoSa \
    --device cuda \
    --dtype float16 \
    --epochs 1 \
    --batch-size 8 \
    --eval-batch-size 8 

  # Multi-GPU DDP training. Match --gpu-ids length with nproc_per_node.
  torchrun --nproc_per_node=2 scripts/run_nugget_experiment.py \
    --config configs/dna_nugget_quick.json \
    --mode train \
    --device cuda \
    --gpu-ids 0 1 \
    --nugget-backbone bart \
    --nugget-tokenizer dnagpt_kmer \
    --variant dna_gpt0.1b_m \
    --arithmetic-coding-mode base_prefix_exact_gpu_cpu \
    --arithmetic-merge-size 2 \
    --batch-size 8 \
    --eval-batch-size 8 \
    --num-workers 8 \
    --prefetch-factor 4 \
    --persistent-workers \
    --pin-memory
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.config import load_experiment_config
from dna_compress.nugget_compression import NUGGET_ARITHMETIC_CODING_MODES
from dna_compress.nugget_experiment import run_nugget_experiment, validate_nugget_config
from dna_compress.nugget_loader import NUGGET_BACKBONES
from dna_compress.nugget_tokenization import NUGGET_TOKENIZERS, apply_nugget_tokenizer_to_model_config, build_nugget_tokenizer_spec


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith("[") and value.endswith("]")) or (value.startswith("{") and value.endswith("}")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _set_nested_attr(config: Any, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) < 2:
        raise ValueError(f"Override key must include section and field: {dotted_key}")
    target = config
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown config section/path: {'.'.join(parts[:-1])}")
        target = getattr(target, part)
    if not hasattr(target, parts[-1]):
        raise ValueError(f"Unknown config field: {dotted_key}")
    setattr(target, parts[-1], value)


def _apply_if_not_none(config: Any, dotted_key: str, value: Any) -> None:
    if value is not None:
        _set_nested_attr(config, dotted_key, value)


def _apply_overrides(config: Any, args: argparse.Namespace) -> None:
    if args.species is not None:
        config.data.species = args.species
    _apply_if_not_none(config, "model.implementation", args.implementation)
    _apply_if_not_none(config, "model.variant", args.variant)
    _apply_if_not_none(config, "model.pretrained_weight_path", args.pretrained_weight_path)
    _apply_if_not_none(config, "model.seq_length", args.seq_length)
    _apply_if_not_none(config, "model.nugget_backbone", args.nugget_backbone)
    _apply_if_not_none(config, "model.nugget_ratio", args.nugget_ratio)
    _apply_if_not_none(config, "model.nugget_scorer_layer", args.nugget_scorer_layer)
    _apply_if_not_none(config, "model.nugget_residual_start", args.nugget_residual_start)
    _apply_if_not_none(config, "model.nugget_residual_end", args.nugget_residual_end)

    _apply_if_not_none(config, "data.dataset_dir", args.dataset_dir)
    _apply_if_not_none(config, "data.nugget_tokenizer", args.nugget_tokenizer)
    _apply_if_not_none(config, "data.train_ratio", args.train_ratio)
    _apply_if_not_none(config, "data.val_ratio", args.val_ratio)
    _apply_if_not_none(config, "data.test_ratio", args.test_ratio)
    _apply_if_not_none(config, "data.max_train_bytes_per_species", args.max_train_bytes)
    _apply_if_not_none(config, "data.max_val_bytes_per_species", args.max_val_bytes)
    _apply_if_not_none(config, "data.max_test_bytes_per_species", args.max_test_bytes)
    _apply_if_not_none(config, "data.train_samples_per_epoch", args.train_samples_per_epoch)
    _apply_if_not_none(config, "data.train_sampling_strategy", args.train_sampling_strategy)
    _apply_if_not_none(config, "data.token_merge_size", args.token_merge_size)
    _apply_if_not_none(config, "data.token_merge_alphabet", args.token_merge_alphabet)
    _apply_if_not_none(config, "data.species_prefix_map", args.species_prefix_map)
    _apply_if_not_none(config, "data.compression_sample_bytes", args.compression_sample_bytes)
    _apply_if_not_none(config, "arithmetic.coding_mode", args.arithmetic_coding_mode)
    _apply_if_not_none(config, "arithmetic.merge_size", args.arithmetic_merge_size)

    for field in (
        "seed",
        "device",
        "gpu_ids",
        "dtype",
        "init_from",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "weight_decay",
        "lr_scheduler",
        "lr_warmup_steps",
        "lr_min_ratio",
        "grad_clip_norm",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "pin_memory",
        "log_interval",
        "eval_interval",
    ):
        _apply_if_not_none(config, f"train.{field}", getattr(args, field))

    for field in (
        "run_name",
        "output_dir",
        "wandb_project",
        "wandb_entity",
        "wandb_name",
        "wandb_group",
        "wandb_tags",
        "wandb_mode",
    ):
        _apply_if_not_none(config, f"output.{field}", getattr(args, field))
    if args.wandb_enabled is not None:
        config.output.wandb_enabled = args.wandb_enabled
    elif args.wandb_project is not None:
        config.output.wandb_enabled = True

    for item in args.override:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected section.key=value")
        key, raw_value = item.split("=", 1)
        _set_nested_attr(config, key.strip(), _parse_scalar(raw_value.strip()))


def _apply_timestamp_to_output_dir(config: Any, args: argparse.Namespace) -> None:
    if args.timestamp_output:
        config.output.output_dir = f"{config.output.output_dir}_{datetime.now().strftime(args.timestamp_format)}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate Nugget on DNACorpus.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="all", choices=["train", "eval", "all"])
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timestamp-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timestamp-format", default="%Y%m%d_%H%M%S")
    parser.add_argument("--override", action="append", default=[])

    model_group = parser.add_argument_group("model overrides")
    model_group.add_argument("--implementation", choices=["nugget"])
    model_group.add_argument("--variant", choices=["dna_gpt0.1b_h", "dna_gpt0.1b_m", "dna_gpt3b_m"])
    model_group.add_argument("--pretrained-weight-path")
    model_group.add_argument("--seq-length", type=int)
    model_group.add_argument("--nugget-backbone", choices=list(NUGGET_BACKBONES))
    model_group.add_argument("--nugget-ratio", type=float)
    model_group.add_argument("--nugget-scorer-layer", type=int)
    model_group.add_argument("--nugget-residual-start", type=int)
    model_group.add_argument("--nugget-residual-end", type=int)

    data_group = parser.add_argument_group("data overrides")
    data_group.add_argument("--dataset-dir")
    data_group.add_argument("--species", nargs="+")
    data_group.add_argument("--nugget-tokenizer", choices=list(NUGGET_TOKENIZERS))
    data_group.add_argument("--species-prefix-map", type=json.loads)
    data_group.add_argument("--train-ratio", type=float)
    data_group.add_argument("--val-ratio", type=float)
    data_group.add_argument("--test-ratio", type=float)
    data_group.add_argument("--max-train-bytes", type=int)
    data_group.add_argument("--max-val-bytes", type=int)
    data_group.add_argument("--max-test-bytes", type=int)
    data_group.add_argument("--train-samples-per-epoch", type=int)
    data_group.add_argument("--train-sampling-strategy", choices=["proportional", "uniform", "sqrt"])
    data_group.add_argument("--token-merge-size", type=int)
    data_group.add_argument("--token-merge-alphabet")
    data_group.add_argument("--compression-sample-bytes", type=int)

    arithmetic_group = parser.add_argument_group("arithmetic overrides")
    arithmetic_group.add_argument("--arithmetic-coding-mode", choices=list(NUGGET_ARITHMETIC_CODING_MODES))
    arithmetic_group.add_argument("--arithmetic-merge-size", type=int)

    train_group = parser.add_argument_group("train overrides")
    train_group.add_argument("--seed", type=int)
    train_group.add_argument("--device")
    train_group.add_argument("--gpu-ids", type=int, nargs="+")
    train_group.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    train_group.add_argument("--init-from", choices=["scratch", "pretrained", "resume"])
    train_group.add_argument("--epochs", type=int)
    train_group.add_argument("--batch-size", type=int)
    train_group.add_argument("--eval-batch-size", type=int)
    train_group.add_argument("--learning-rate", type=float)
    train_group.add_argument("--weight-decay", type=float)
    train_group.add_argument("--lr-scheduler", choices=["none", "linear", "cosine"])
    train_group.add_argument("--lr-warmup-steps", type=int)
    train_group.add_argument("--lr-min-ratio", type=float)
    train_group.add_argument("--grad-clip-norm", type=float)
    train_group.add_argument("--num-workers", type=int)
    train_group.add_argument("--prefetch-factor", type=int)
    train_group.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None)
    train_group.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    train_group.add_argument("--log-interval", type=int)
    train_group.add_argument("--eval-interval", type=int)

    output_group = parser.add_argument_group("output overrides")
    output_group.add_argument("--run-name")
    output_group.add_argument("--output-dir")
    output_group.add_argument("--wandb-project")
    output_group.add_argument("--wandb-entity")
    output_group.add_argument("--wandb-name")
    output_group.add_argument("--wandb-group")
    output_group.add_argument("--wandb-tags", nargs="+")
    output_group.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    output_group.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_experiment_config(args.config)
    _apply_overrides(config, args)
    _apply_timestamp_to_output_dir(config, args)
    tokenizer_spec = build_nugget_tokenizer_spec(config.data, config.model)
    apply_nugget_tokenizer_to_model_config(config.model, tokenizer_spec)
    validate_nugget_config(config)
    if args.print_config or args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    if args.dry_run:
        print("Dry-run completed: Nugget config resolved and validated.")
        return
    print(json.dumps(run_nugget_experiment(config, mode=args.mode), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
