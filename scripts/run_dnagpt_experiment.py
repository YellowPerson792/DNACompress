from __future__ import annotations

"""Train or evaluate DNAGPT on DNACorpus with config + CLI overrides.

Example: warm-start DNAGPT 0.1B multi-organism on HoSa using GPU 3

python scripts/run_dnagpt_experiment.py \
        --config configs/dna_dnagpt_quick.json \
        --mode all \
        --variant dna_gpt0.1b_m \
        --pretrained-weight-path third_party/DNAGPT/checkpoints/dna_gpt0.1b_m.pth \
        --init-from pretrained \
        --dtype bfloat16 \
        --epochs 1 \
        --batch-size 32 \
        --eval-batch-size 32 \
        --learning-rate 3e-4 \
        --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
        --train-samples-per-epoch 600000 \
        --compression-sample-bytes 100000 \
        --print-config \
        --seq-length 512 \
        --weight-decay 0.01 \
        --log-interval 25 \
        --eval-interval 500 \
        --train-ratio 0.6 \
        --val-ratio 0.2 \
        --test-ratio 0.2 \
        --lr-scheduler cosine \
        --lr-warmup-steps 0 \
        --lr-min-ratio 0.1 \
        --grad-clip-norm 1.0 \
        --num-workers 4 \
        --train-sampling-strategy proportional 
            
        --gpu-ids 1 3 \
        --wandb-project dna-compress \
        --wandb-name dnagpt-realtime \
        --wandb-project dna-compress 
        
Example: Multi-GPU training

    torchrun --nproc_per_node=2 scripts/run_dnagpt_experiment.py \
        --config configs/dna_dnagpt_quick.json \
        --dry-run \
        --no-timestamp-output
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
from dna_compress.dnagpt_experiment import run_dnagpt_experiment, validate_dnagpt_config


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

    target: Any = config
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown config section/path: {'.'.join(parts[:-1])}")
        target = getattr(target, part)

    field_name = parts[-1]
    if not hasattr(target, field_name):
        raise ValueError(f"Unknown config field: {dotted_key}")
    setattr(target, field_name, value)


def _apply_if_not_none(config: Any, dotted_key: str, value: Any) -> None:
    if value is None:
        return
    _set_nested_attr(config, dotted_key, value)


def _apply_overrides(config: Any, args: argparse.Namespace) -> None:
    if args.species is not None:
        config.data.species = args.species

    _apply_if_not_none(config, "model.implementation", args.implementation)
    _apply_if_not_none(config, "model.variant", args.variant)
    _apply_if_not_none(config, "model.pretrained_weight_path", args.pretrained_weight_path)
    _apply_if_not_none(config, "model.seq_length", args.seq_length)

    _apply_if_not_none(config, "data.dataset_dir", args.dataset_dir)
    _apply_if_not_none(config, "data.train_ratio", args.train_ratio)
    _apply_if_not_none(config, "data.val_ratio", args.val_ratio)
    _apply_if_not_none(config, "data.test_ratio", args.test_ratio)
    _apply_if_not_none(config, "data.max_train_bytes_per_species", args.max_train_bytes)
    _apply_if_not_none(config, "data.max_val_bytes_per_species", args.max_val_bytes)
    _apply_if_not_none(config, "data.max_test_bytes_per_species", args.max_test_bytes)
    _apply_if_not_none(config, "data.train_samples_per_epoch", args.train_samples_per_epoch)
    _apply_if_not_none(config, "data.train_sampling_strategy", args.train_sampling_strategy)
    _apply_if_not_none(config, "data.species_prefix_map", args.species_prefix_map)
    _apply_if_not_none(config, "data.compression_sample_bytes", args.compression_sample_bytes)

    _apply_if_not_none(config, "train.seed", args.seed)
    _apply_if_not_none(config, "train.device", args.device)
    _apply_if_not_none(config, "train.gpu_ids", args.gpu_ids)
    _apply_if_not_none(config, "train.dtype", args.dtype)
    _apply_if_not_none(config, "train.init_from", args.init_from)
    _apply_if_not_none(config, "train.epochs", args.epochs)
    _apply_if_not_none(config, "train.batch_size", args.batch_size)
    _apply_if_not_none(config, "train.eval_batch_size", args.eval_batch_size)
    _apply_if_not_none(config, "train.learning_rate", args.learning_rate)
    _apply_if_not_none(config, "train.weight_decay", args.weight_decay)
    _apply_if_not_none(config, "train.lr_scheduler", args.lr_scheduler)
    _apply_if_not_none(config, "train.lr_warmup_steps", args.lr_warmup_steps)
    _apply_if_not_none(config, "train.lr_min_ratio", args.lr_min_ratio)
    _apply_if_not_none(config, "train.grad_clip_norm", args.grad_clip_norm)
    _apply_if_not_none(config, "train.num_workers", args.num_workers)
    _apply_if_not_none(config, "train.prefetch_factor", args.prefetch_factor)
    _apply_if_not_none(config, "train.persistent_workers", args.persistent_workers)
    _apply_if_not_none(config, "train.pin_memory", args.pin_memory)
    _apply_if_not_none(config, "train.log_interval", args.log_interval)
    _apply_if_not_none(config, "train.eval_interval", args.eval_interval)

    _apply_if_not_none(config, "output.run_name", args.run_name)
    _apply_if_not_none(config, "output.output_dir", args.output_dir)
    _apply_if_not_none(config, "output.wandb_project", args.wandb_project)
    _apply_if_not_none(config, "output.wandb_entity", args.wandb_entity)
    _apply_if_not_none(config, "output.wandb_name", args.wandb_name)
    _apply_if_not_none(config, "output.wandb_group", args.wandb_group)
    _apply_if_not_none(config, "output.wandb_tags", args.wandb_tags)
    _apply_if_not_none(config, "output.wandb_mode", args.wandb_mode)
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
    if not args.timestamp_output:
        return
    timestamp = datetime.now().strftime(args.timestamp_format)
    config.output.output_dir = f"{config.output.output_dir}_{timestamp}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate DNAGPT on DNACorpus with config + CLI overrides.")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    parser.add_argument("--mode", default="all", choices=["train", "eval", "all"])
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--timestamp-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append timestamp suffix to output_dir (default: enabled). Use --no-timestamp-output to disable.",
    )
    parser.add_argument("--timestamp-format", default="%Y%m%d_%H%M%S")
    parser.add_argument("--override", action="append", default=[], help="Generic override in form section.key=value.")

    model_group = parser.add_argument_group("model overrides")
    model_group.add_argument("--implementation", choices=["dnagpt"])
    model_group.add_argument("--variant", choices=["dna_gpt0.1b_h", "dna_gpt0.1b_m", "dna_gpt3b_m"])
    model_group.add_argument("--pretrained-weight-path")
    model_group.add_argument("--seq-length", type=int, help="DNAGPT sequence length measured in tokens.")

    data_group = parser.add_argument_group("data overrides")
    data_group.add_argument("--dataset-dir")
    data_group.add_argument("--species", nargs="+")
    data_group.add_argument("--species-prefix-map", type=json.loads, help='JSON dict, e.g. {"OrSa":"R"}')
    data_group.add_argument("--train-ratio", type=float)
    data_group.add_argument("--val-ratio", type=float)
    data_group.add_argument("--test-ratio", type=float)
    data_group.add_argument("--max-train-bytes", type=int)
    data_group.add_argument("--max-val-bytes", type=int)
    data_group.add_argument("--max-test-bytes", type=int)
    data_group.add_argument("--train-samples-per-epoch", type=int)
    data_group.add_argument("--train-sampling-strategy", choices=["proportional", "uniform", "sqrt"])
    data_group.add_argument("--compression-sample-bytes", type=int)

    train_group = parser.add_argument_group("train overrides")
    train_group.add_argument("--seed", type=int)
    train_group.add_argument("--device")
    train_group.add_argument("--gpu-ids", type=int, nargs="+", help="Preferred CUDA device IDs, e.g. --gpu-ids 1 3")
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
    output_group.add_argument("--wandb-project", help="Enable realtime W&B logging and set project name.")
    output_group.add_argument("--wandb-entity", help="Optional W&B entity/team.")
    output_group.add_argument("--wandb-name", help="Optional W&B run name.")
    output_group.add_argument("--wandb-group", help="Optional W&B group.")
    output_group.add_argument("--wandb-tags", nargs="+", help="Optional W&B tags.")
    output_group.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    output_group.add_argument(
        "--wandb-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force enable/disable realtime W&B logging.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    _apply_overrides(config, args)
    _apply_timestamp_to_output_dir(config, args)
    validate_dnagpt_config(config)

    if args.print_config or args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry-run completed: config resolved and validated.")
        return

    metrics = run_dnagpt_experiment(config, mode=args.mode)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
