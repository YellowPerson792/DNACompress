from __future__ import annotations

"""Run DNA Megabyte experiments.

Complete example (train + eval + compression, with common overrides):

    python scripts/run_dna_experiment.py \
      --config configs/dna_megabyte_quick.json \
      --mode all \
      --dtype float16 \
      --epochs 1 \
      --batch-size 16 \
      --eval-batch-size 32 \
      --learning-rate 3e-4 \
      --weight-decay 0.01 \
      --species HoSa \
      --train-samples-per-epoch 50000 \
      --compression-sample-bytes 16384 \
      --output-dir outputs/dna_megabyte \
      --print-config
    
      --seq-length 1024 \
      --patch-size 4 \
          
Optional generic overrides (repeatable):

    --override train.epochs=2 --override data.species='["HoSa","YeMi"]'
"""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress import load_experiment_config, run_experiment


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
    # Generic overrides take top priority after explicit CLI flags.
    if args.species is not None:
        config.data.species = args.species

    _apply_if_not_none(config, "model.implementation", args.implementation)
    _apply_if_not_none(config, "model.patch_size", args.patch_size)
    _apply_if_not_none(config, "model.seq_length", args.seq_length)
    _apply_if_not_none(config, "model.global_dim", args.global_dim)
    _apply_if_not_none(config, "model.local_dim", args.local_dim)
    _apply_if_not_none(config, "model.global_heads", args.global_heads)
    _apply_if_not_none(config, "model.global_layers", args.global_layers)
    _apply_if_not_none(config, "model.local_heads", args.local_heads)
    _apply_if_not_none(config, "model.local_layers", args.local_layers)

    _apply_if_not_none(config, "data.dataset_dir", args.dataset_dir)
    _apply_if_not_none(config, "data.train_ratio", args.train_ratio)
    _apply_if_not_none(config, "data.val_ratio", args.val_ratio)
    _apply_if_not_none(config, "data.test_ratio", args.test_ratio)
    _apply_if_not_none(config, "data.max_train_bytes_per_species", args.max_train_bytes)
    _apply_if_not_none(config, "data.max_val_bytes_per_species", args.max_val_bytes)
    _apply_if_not_none(config, "data.max_test_bytes_per_species", args.max_test_bytes)
    _apply_if_not_none(config, "data.train_samples_per_epoch", args.train_samples_per_epoch)
    _apply_if_not_none(config, "data.compression_sample_bytes", args.compression_sample_bytes)

    _apply_if_not_none(config, "train.seed", args.seed)
    _apply_if_not_none(config, "train.device", args.device)
    _apply_if_not_none(config, "train.dtype", args.dtype)
    _apply_if_not_none(config, "train.epochs", args.epochs)
    _apply_if_not_none(config, "train.batch_size", args.batch_size)
    _apply_if_not_none(config, "train.eval_batch_size", args.eval_batch_size)
    _apply_if_not_none(config, "train.learning_rate", args.learning_rate)
    _apply_if_not_none(config, "train.weight_decay", args.weight_decay)
    _apply_if_not_none(config, "train.grad_clip_norm", args.grad_clip_norm)
    _apply_if_not_none(config, "train.num_workers", args.num_workers)
    _apply_if_not_none(config, "train.log_interval", args.log_interval)
    _apply_if_not_none(config, "train.eval_interval", args.eval_interval)

    _apply_if_not_none(config, "output.run_name", args.run_name)
    _apply_if_not_none(config, "output.output_dir", args.output_dir)

    for item in args.override:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected section.key=value")
        key, raw_value = item.split("=", 1)
        _set_nested_attr(config, key.strip(), _parse_scalar(raw_value.strip()))


def _validate_config_for_megabyte(config: Any) -> None:
    if config.model.implementation not in {"megabyte", "megabyte_in_action"}:
        raise ValueError(
            "model.implementation must be one of 'megabyte' or 'megabyte_in_action' "
            f"for this project, got '{config.model.implementation}'."
        )

    if config.model.seq_length <= 0 or config.model.patch_size <= 0:
        raise ValueError("model.seq_length and model.patch_size must be > 0")

    if config.model.seq_length % config.model.patch_size != 0:
        raise ValueError(
            f"model.seq_length ({config.model.seq_length}) must be divisible by "
            f"model.patch_size ({config.model.patch_size}) for Megabyte."
        )

    ratio_sum = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"data split ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"(train={config.data.train_ratio}, val={config.data.val_ratio}, test={config.data.test_ratio})."
        )

    if config.train.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("train.dtype must be one of: float32, float16, bfloat16")

    if config.train.batch_size <= 0 or config.train.eval_batch_size <= 0:
        raise ValueError("train.batch_size and train.eval_batch_size must be > 0")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/eval/compress Megabyte on DNACorpus with config + CLI overrides.",
        epilog=(
            "Examples:\n"
            "  python scripts/run_dna_experiment.py --config configs/dna_megabyte_quick.json --mode all\n"
            "  python scripts/run_dna_experiment.py --config configs/dna_megabyte_quick.json --mode train "
            "--batch-size 8 --learning-rate 1e-4 --seq-length 512\n"
            "  python scripts/run_dna_experiment.py --config configs/dna_megabyte_quick.json "
            "--override train.epochs=2 --override data.species='[\"HoSa\",\"YeMi\"]'"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["train", "eval", "compress", "all"],
        help="Which stage to run.",
    )
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config before running.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate config, then exit.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Generic override in form section.key=value. Can be repeated.",
    )

    model_group = parser.add_argument_group("model overrides")
    model_group.add_argument("--implementation", choices=["megabyte", "megabyte_in_action"])
    model_group.add_argument("--patch-size", type=int)
    model_group.add_argument("--seq-length", type=int)
    model_group.add_argument("--global-dim", type=int)
    model_group.add_argument("--local-dim", type=int)
    model_group.add_argument("--global-heads", type=int)
    model_group.add_argument("--global-layers", type=int)
    model_group.add_argument("--local-heads", type=int)
    model_group.add_argument("--local-layers", type=int)

    data_group = parser.add_argument_group("data overrides")
    data_group.add_argument("--dataset-dir")
    data_group.add_argument("--species", nargs="+", help="Species list, e.g. --species HoSa YeMi")
    data_group.add_argument("--train-ratio", type=float)
    data_group.add_argument("--val-ratio", type=float)
    data_group.add_argument("--test-ratio", type=float)
    data_group.add_argument("--max-train-bytes", type=int)
    data_group.add_argument("--max-val-bytes", type=int)
    data_group.add_argument("--max-test-bytes", type=int)
    data_group.add_argument("--train-samples-per-epoch", type=int)
    data_group.add_argument("--compression-sample-bytes", type=int)

    train_group = parser.add_argument_group("train overrides")
    train_group.add_argument("--seed", type=int)
    train_group.add_argument("--device", help="auto/cpu/cuda/cuda:0 ...")
    train_group.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    train_group.add_argument("--epochs", type=int)
    train_group.add_argument("--batch-size", type=int)
    train_group.add_argument("--eval-batch-size", type=int)
    train_group.add_argument("--learning-rate", type=float)
    train_group.add_argument("--weight-decay", type=float)
    train_group.add_argument("--grad-clip-norm", type=float)
    train_group.add_argument("--num-workers", type=int)
    train_group.add_argument("--log-interval", type=int)
    train_group.add_argument("--eval-interval", type=int)

    output_group = parser.add_argument_group("output overrides")
    output_group.add_argument("--run-name")
    output_group.add_argument("--output-dir")

    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    _apply_overrides(config, args)
    _validate_config_for_megabyte(config)

    if args.print_config or args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry-run completed: config resolved and validated.")
        return

    metrics = run_experiment(config, mode=args.mode)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
