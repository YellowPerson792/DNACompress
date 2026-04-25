from __future__ import annotations

"""Run DNA Megabyte experiments.

1x train-split coverage reference for `seq_length=1024`
(`train_ratio=0.9`, `max_train_bytes_per_species=null`).
Species shown in the reference figure are listed first in figure order; local-only
datasets are appended at the end.

| Species | Train bytes | `train_samples_per_epoch` for 1x coverage |
| --- | ---: | ---: |
| OrSa | 38,936,271 | 38,024 |
| HoSa | 170,777,400 | 166,775 |
| DaRe | 56,308,518 | 54,989 |
| ScPo | 9,586,940 | 9,363 |
| EsCo | 4,177,487 | 4,080 |
| YeMi | 66,320 | 65 |
| BuEb | 17,046 | 17 |
| AgPh | 39,573 | 39 |
| Total (through AgPh) | 273,352,572 | 273,352 |
| GaGa | 133,679,065 | 130,546 |
| DrMe | 28,963,286 | 28,285 |
| EnIn | 23,762,778 | 23,206 |
| PlFa | 8,088,041 | 7,899 |
| HePy | 1,501,042 | 1,466 |
| AeCa | 1,431,944 | 1,399 |
| HaHi | 3,501,004 | 3,419 |
| AnCa | 127,970,708 | 124,972 |
| WaMe | 8,229,989 | 8,038 |
| Total (all local datasets) | 617,044,512 | 602,582 |

Complete example (train + eval + compression, with common overrides):

    python scripts/run_dna_experiment.py \
        --config configs/dna_megabyte_large.json \
        --mode all \
        --init-from scratch \
        --pretrained-weight-path outputs/dna_megabyte_huge_ensembl_all/best.pt \
        --seed 42 \
        --dataset-dir datasets/DNACorpus \
        --sequence-source-mode auto \
        --multi-sequence-mode separate \
        --dtype bfloat16 \
        --epochs 1 \
        --batch-size 32 \
        --eval-batch-size 32 \
        --learning-rate 1e-4 \
        --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
        --train-samples-per-epoch 600000 \
        --compression-sample-bytes 100000 \
        --print-config \
        --seq-length 1024 \
        --token-merge-size 3 \
        --weight-decay 0.01 \
        --log-interval 25 \
        --eval-interval 2500 \
        --train-ratio 0.6 \
        --val-ratio 0.2 \
        --test-ratio 0.2 \
        --lr-scheduler cosine \
        --lr-warmup-steps 0 \
        --lr-min-ratio 0.1 \
        --grad-clip-norm 1.0 \
        --num-workers 4 \
        --train-sampling-strategy proportional 
            
        --wandb-project dna-compress \
        --wandb-name dna_megabyte_huge_ensembl_all_resume \
        --gpu-ids 2 3 
        
        --species homo_sapiens mus_musculus bos_taurus danio_rerio \
                  drosophila_melanogaster caenorhabditis_elegans \
                  saccharomyces_cerevisiae arabidopsis_thaliana \
        --wandb-project dna-compress \
        --wandb-name dna_megabyte_huge_ensembl_all \
        --gpu-ids 0 3 \
        --init-from pretrained \
        --pretrained-weight-path outputs/dna_megabyte_huge_b128_ensembl_all/best.pt \
        --input-causal-conv-kernel-size 7

Multi-GPU DDP example (2 GPUs):

    torchrun --nproc_per_node=2 scripts/run_dna_experiment.py \
        --config configs/dna_megabyte_quick.json \
        --mode train \
        --dataset-dir datasets/ensembl_raw \
        --sequence-source-mode auto \
        --multi-sequence-mode separate \
        --species homo_sapiens mus_musculus bos_taurus danio_rerio \
                  drosophila_melanogaster caenorhabditis_elegans \
                  saccharomyces_cerevisiae arabidopsis_thaliana \
        --device cuda \
        --num-workers 8 \
        --prefetch-factor 4 \
        --persistent-workers \
        --pin-memory \
        --gpus 0 1
    
          
Optional generic overrides (repeatable):

    --override train.epochs=2 --override data.species='["GaGa","DrMe"]'
"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.config import load_experiment_config
from dna_compress.tokenization import apply_token_merge_to_model_config, normalize_alphabet


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


def _parse_gpu_ids(values: list[str]) -> list[int]:
    gpu_ids: list[int] = []
    for value in values:
        for token in value.split(","):
            item = token.strip()
            if not item:
                continue
            try:
                gpu_id = int(item)
            except ValueError as error:
                raise ValueError(f"Invalid GPU id '{item}'. Expected integers like 0 1 or 0,1.") from error
            if gpu_id < 0:
                raise ValueError(f"Invalid GPU id '{item}'. GPU id must be >= 0.")
            gpu_ids.append(gpu_id)

    if not gpu_ids:
        raise ValueError("--gpus was provided but no valid GPU ids were parsed.")

    # Preserve order while dropping duplicates.
    deduplicated_gpu_ids: list[int] = []
    for gpu_id in gpu_ids:
        if gpu_id not in deduplicated_gpu_ids:
            deduplicated_gpu_ids.append(gpu_id)
    return deduplicated_gpu_ids


def _parse_sequence_include(values: list[str] | None) -> dict[str, list[str]] | None:
    if values is None:
        return None
    parsed: dict[str, list[str]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid sequence include '{item}'. Expected species=key1,key2,...")
        species, raw_keys = item.split("=", 1)
        species_name = species.strip()
        keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
        if not species_name or not keys:
            raise ValueError(f"Invalid sequence include '{item}'. Expected species=key1,key2,...")
        parsed.setdefault(species_name, [])
        for key in keys:
            if key not in parsed[species_name]:
                parsed[species_name].append(key)
    return parsed


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
    _apply_if_not_none(config, "model.pretrained_weight_path", args.pretrained_weight_path)
    _apply_if_not_none(config, "model.patch_size", args.patch_size)
    _apply_if_not_none(config, "model.seq_length", args.seq_length)
    _apply_if_not_none(config, "model.global_dim", args.global_dim)
    _apply_if_not_none(config, "model.local_dim", args.local_dim)
    _apply_if_not_none(config, "model.global_heads", args.global_heads)
    _apply_if_not_none(config, "model.global_layers", args.global_layers)
    _apply_if_not_none(config, "model.local_heads", args.local_heads)
    _apply_if_not_none(config, "model.local_layers", args.local_layers)
    _apply_if_not_none(config, "model.attn_dropout", args.attn_dropout)
    _apply_if_not_none(config, "model.ff_dropout", args.ff_dropout)
    _apply_if_not_none(config, "model.input_causal_conv_kernel_size", args.input_causal_conv_kernel_size)

    _apply_if_not_none(config, "data.dataset_dir", args.dataset_dir)
    _apply_if_not_none(config, "data.sequence_source_mode", args.sequence_source_mode)
    _apply_if_not_none(config, "data.multi_sequence_mode", args.multi_sequence_mode)
    _apply_if_not_none(config, "data.clean_cache_enabled", args.clean_cache_enabled)
    _apply_if_not_none(config, "data.clean_cache_dir", args.clean_cache_dir)
    if args.sequence_include is not None:
        config.data.sequence_include_map = _parse_sequence_include(args.sequence_include)
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
    _apply_if_not_none(config, "data.compression_sample_bytes", args.compression_sample_bytes)

    _apply_if_not_none(config, "train.seed", args.seed)
    _apply_if_not_none(config, "train.device", args.device)
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
    if args.gpus is not None:
        config.train.gpu_ids = _parse_gpu_ids(args.gpus)

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


def _validate_config_for_megabyte(config: Any) -> None:
    if config.model.implementation not in {
        "megabyte",
        "megabyte_in_action",
        "megabyte_in_action_causal_conv",
        "megabyte_relative",
    }:
        raise ValueError(
            "model.implementation must be one of 'megabyte', 'megabyte_in_action', "
            "'megabyte_in_action_causal_conv', or 'megabyte_relative' "
            f"for this project, got '{config.model.implementation}'."
        )

    if config.model.seq_length <= 0 or config.model.patch_size <= 0:
        raise ValueError("model.seq_length and model.patch_size must be > 0")

    if config.model.seq_length % config.model.patch_size != 0:
        raise ValueError(
            f"model.seq_length ({config.model.seq_length}) must be divisible by "
            f"model.patch_size ({config.model.patch_size}) for Megabyte."
        )

    if not (0.0 <= config.model.attn_dropout < 1.0):
        raise ValueError("model.attn_dropout must be in [0.0, 1.0)")

    if not (0.0 <= config.model.ff_dropout < 1.0):
        raise ValueError("model.ff_dropout must be in [0.0, 1.0)")

    if config.model.input_causal_conv_kernel_size <= 0:
        raise ValueError("model.input_causal_conv_kernel_size must be >= 1")

    ratio_sum = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"data split ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"(train={config.data.train_ratio}, val={config.data.val_ratio}, test={config.data.test_ratio})."
        )

    if config.train.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("train.dtype must be one of: float32, float16, bfloat16")

    if config.train.init_from not in {"scratch", "pretrained", "resume"}:
        raise ValueError("train.init_from must be one of: scratch, pretrained, resume")

    if config.train.batch_size <= 0 or config.train.eval_batch_size <= 0:
        raise ValueError("train.batch_size and train.eval_batch_size must be > 0")

    if config.data.train_sampling_strategy not in {"proportional", "uniform", "sqrt"}:
        raise ValueError("data.train_sampling_strategy must be one of: proportional, uniform, sqrt")

    if config.data.token_merge_size <= 0:
        raise ValueError("data.token_merge_size must be >= 1")

    normalize_alphabet(config.data.token_merge_alphabet)
    if config.data.sequence_source_mode not in {"auto", "flat_file", "fasta_dir"}:
        raise ValueError("data.sequence_source_mode must be one of: auto, flat_file, fasta_dir")
    if config.data.multi_sequence_mode not in {"separate", "concat"}:
        raise ValueError("data.multi_sequence_mode must be one of: separate, concat")
    if not isinstance(config.data.sequence_include_map, dict):
        raise ValueError("data.sequence_include_map must be a dict[str, list[str]]")
    for species_name, keys in config.data.sequence_include_map.items():
        if not isinstance(species_name, str) or not species_name:
            raise ValueError("data.sequence_include_map keys must be non-empty strings")
        if not isinstance(keys, list) or not keys or any((not isinstance(key, str) or not key.strip()) for key in keys):
            raise ValueError(f"data.sequence_include_map[{species_name!r}] must be a non-empty list of strings")

    if config.train.lr_scheduler not in {"none", "linear", "cosine"}:
        raise ValueError("train.lr_scheduler must be one of: none, linear, cosine")

    if config.train.lr_warmup_steps < 0:
        raise ValueError("train.lr_warmup_steps must be >= 0")

    if not (0.0 <= config.train.lr_min_ratio <= 1.0):
        raise ValueError("train.lr_min_ratio must be in [0.0, 1.0]")

    if config.train.num_workers < 0:
        raise ValueError("train.num_workers must be >= 0")

    if config.train.prefetch_factor <= 0:
        raise ValueError("train.prefetch_factor must be >= 1")
    if config.arithmetic.frequency_total is not None and config.arithmetic.frequency_total <= 0:
        raise ValueError("arithmetic.frequency_total must be > 0 when provided")
    if not (0.0 < config.arithmetic.target_uniform_mass <= 1.0):
        raise ValueError("arithmetic.target_uniform_mass must be in (0.0, 1.0]")

    if config.train.gpu_ids is not None:
        if len(config.train.gpu_ids) == 0:
            raise ValueError("train.gpu_ids cannot be an empty list when provided")
        if any((not isinstance(gpu_id, int) or gpu_id < 0) for gpu_id in config.train.gpu_ids):
            raise ValueError("train.gpu_ids must be a list of non-negative integers")


def _apply_timestamp_to_output_dir(config: Any, args: argparse.Namespace) -> None:
    if not args.timestamp_output:
        return
    if config.train.init_from == "resume" and not config.model.pretrained_weight_path:
        return
    timestamp = datetime.now().strftime(args.timestamp_format)
    config.output.output_dir = f"{config.output.output_dir}_{timestamp}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/eval/compress Megabyte on DNACorpus with config + CLI overrides.",
        epilog=(
            "Examples:\n"
            "  python scripts/run_dna_experiment.py --config configs/dna_megabyte_quick.json --mode all\n"
            "  python scripts/run_dna_experiment.py --config configs/dna_megabyte_quick.json --mode train "
            "--batch-size 8 --learning-rate 1e-4 --seq-length 512\n"
            "  torchrun --nproc_per_node=2 scripts/run_dna_experiment.py --config configs/dna_megabyte_quick.json "
            "--mode train --device cuda --gpus 0 1 --num-workers 8 --prefetch-factor 4 --persistent-workers\n"
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
        "--timestamp-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append timestamp suffix to output_dir (default: enabled). Use --no-timestamp-output to disable.",
    )
    parser.add_argument(
        "--timestamp-format",
        default="%Y%m%d_%H%M%S",
        help="strftime format for output timestamp, default: %%Y%%m%%d_%%H%%M%%S",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Generic override in form section.key=value. Can be repeated.",
    )

    model_group = parser.add_argument_group("model overrides")
    model_group.add_argument(
        "--implementation",
        choices=["megabyte", "megabyte_in_action", "megabyte_in_action_causal_conv", "megabyte_relative"],
    )
    model_group.add_argument("--pretrained-weight-path")
    model_group.add_argument("--patch-size", type=int)
    model_group.add_argument("--seq-length", type=int)
    model_group.add_argument("--global-dim", type=int)
    model_group.add_argument("--local-dim", type=int)
    model_group.add_argument("--global-heads", type=int)
    model_group.add_argument("--global-layers", type=int)
    model_group.add_argument("--local-heads", type=int)
    model_group.add_argument("--local-layers", type=int)
    model_group.add_argument("--attn-dropout", type=float, help="Attention dropout used in both global and local transformers.")
    model_group.add_argument("--ff-dropout", type=float, help="Feed-forward dropout used in both global and local transformers.")
    model_group.add_argument(
        "--input-causal-conv-kernel-size",
        type=int,
        help="Kernel size for the causal Conv1d added before Megabyte-in-Action token embeddings are consumed.",
    )

    data_group = parser.add_argument_group("data overrides")
    data_group.add_argument("--dataset-dir")
    data_group.add_argument("--species", nargs="+", help="Species list, e.g. --species HoSa YeMi")
    data_group.add_argument("--sequence-source-mode", choices=["auto", "flat_file", "fasta_dir"])
    data_group.add_argument("--multi-sequence-mode", choices=["separate", "concat"])
    data_group.add_argument("--clean-cache-enabled", dest="clean_cache_enabled", action="store_true", default=None)
    data_group.add_argument("--no-clean-cache", dest="clean_cache_enabled", action="store_false")
    data_group.add_argument("--clean-cache-dir")
    data_group.add_argument(
        "--sequence-include",
        action="append",
        help="Repeatable sequence selector in form species=key1,key2,...",
    )
    data_group.add_argument("--train-ratio", type=float)
    data_group.add_argument("--val-ratio", type=float)
    data_group.add_argument("--test-ratio", type=float)
    data_group.add_argument("--max-train-bytes", type=int)
    data_group.add_argument("--max-val-bytes", type=int)
    data_group.add_argument("--max-test-bytes", type=int)
    data_group.add_argument("--train-samples-per-epoch", type=int)
    data_group.add_argument("--train-sampling-strategy", choices=["proportional", "uniform", "sqrt"])
    data_group.add_argument("--token-merge-size", type=int, help="Merge this many DNA bases into one token. 1 keeps byte-level tokens.")
    data_group.add_argument("--token-merge-alphabet", help="DNA alphabet used for merged-token encoding, e.g. ACGTN")
    data_group.add_argument("--compression-sample-bytes", type=int)

    train_group = parser.add_argument_group("train overrides")
    train_group.add_argument("--seed", type=int)
    train_group.add_argument("--device", help="auto/cpu/cuda/cuda:0 ...")
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
    train_group.add_argument("--prefetch-factor", type=int, help="DataLoader prefetch factor per worker (effective when num_workers > 0).")
    train_group.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep DataLoader workers alive between epochs (effective when num_workers > 0).",
    )
    train_group.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable pinned host memory for faster host-to-device transfer on CUDA.",
    )
    train_group.add_argument("--log-interval", type=int)
    train_group.add_argument("--eval-interval", type=int)
    train_group.add_argument(
        "--gpus",
        "--gpu-ids",
        nargs="+",
        help="GPU ids to use, e.g. --gpus 0 1 or --gpus 0,1. For DDP, launch with torchrun and set nproc_per_node to len(gpu_ids).",
    )

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
    _validate_config_for_megabyte(config)
    apply_token_merge_to_model_config(config.model, config.data)

    if args.print_config or args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry-run completed: config resolved and validated.")
        return

    print("[startup] importing training runtime...", flush=True)
    from dna_compress.experiment import run_experiment

    metrics = run_experiment(config, mode=args.mode)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
