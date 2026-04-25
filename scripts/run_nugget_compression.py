from __future__ import annotations

"""Run standalone Nugget compression on DNACorpus.

Examples:

  # Compress the test split from a trained run using the saved resolved config.
  python scripts/run_nugget_compression.py \
    --run-dir outputs/dna_nugget_bart_byte \
    --checkpoint-tag best \
    --split test \
    --compression-modes windows_nonoverlap \
    --device cuda:0 \
    --dtype float16 \
    --eval-batch-size 8 \
    --compression-sample-bytes 60000 \
    --nugget-hidden-mode runtime_hidden \
    --nugget-hidden-storage-dtype runtime \
    --arithmetic-coding-mode model_symbol \
    --arithmetic-merge-size 1 

  # Stored-hidden compression: cast nuggets to fp16, then decode from fp16 payload.
  python scripts/run_nugget_compression.py \
    --run-dir outputs/dna_nugget_bart_byte \
    --checkpoint-tag best \
    --split train val test \
    --compression-modes windows_nonoverlap \
    --dataset-dir datasets/DNACorpus \
    --species HoSa \
    --train-ratio 0.9 \
    --val-ratio 0.05 \
    --test-ratio 0.05 \
    --device cuda:0 \
    --dtype float16 \
    --eval-batch-size 8 \
    --compression-sample-bytes 60000 \
    --nugget-hidden-mode stored_hidden \
    --nugget-hidden-storage-dtype float16 \
    --arithmetic-coding-mode model_symbol \
    --arithmetic-merge-size 1 \
    --output-json outputs/dna_nugget_bart_byte/nugget_compression_stored_fp16.json

  # Fixed-kmer tokenizer with token-internal arithmetic coding.
  python scripts/run_nugget_compression.py \
    --config configs/dna_nugget_quick.json \
    --checkpoint outputs/dna_nugget_t5_k3_stored_fp16/best.pt \
    --split test \
    --compression-modes windows_nonoverlap \
    --nugget-tokenizer fixed_kmer \
    --token-merge-size 3 \
    --token-merge-alphabet ACGTN \
    --arithmetic-coding-mode fixed_token_units \
    --arithmetic-merge-size 1 \
    --nugget-hidden-mode stored_hidden \
    --nugget-hidden-storage-dtype float16 \
    --device cuda:0 \
    --dtype float16 \
    --eval-batch-size 8 \
    --compression-sample-bytes 60000 \
    --output-json outputs/dna_nugget_t5_k3_stored_fp16/compression_compare.json

  # DNAGPT k-mer tokenizer with base/prefix arithmetic coding.
  python scripts/run_nugget_compression.py \
    --run-dir outputs/dna_nugget_bart_dnagpt_kmer \
    --checkpoint-tag best \
    --split test \
    --compression-modes windows_nonoverlap \
    --nugget-tokenizer dnagpt_kmer \
    --variant dna_gpt0.1b_m \
    --species HoSa \
    --arithmetic-coding-mode base_prefix_exact_gpu_cpu \
    --arithmetic-merge-size 2 \
    --nugget-hidden-mode runtime_hidden \
    --nugget-hidden-storage-dtype runtime \
    --device cuda:0 \
    --dtype float16 \
    --eval-batch-size 8 \
    --compression-sample-bytes 60000
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.compression_eval import NON_OVERLAP_MODE
from dna_compress.config import ExperimentConfig, load_experiment_config
from dna_compress.data import load_splits
from dna_compress.dnagpt_prefix_coding import build_dnagpt_prefix_trie
from dna_compress.experiment import resolve_device
from dna_compress.fixed_token_factorization import build_fixed_token_arithmetic_factorizer
from dna_compress.nugget_compression import (
    NUGGET_ARITHMETIC_CODING_MODES,
    SUPPORTED_NUGGET_COMPRESSION_MODES,
    compress_nugget_source,
    summarize_nugget_per_source,
)
from dna_compress.nugget_experiment import validate_nugget_config
from dna_compress.nugget_loader import (
    NUGGET_HIDDEN_MODES,
    NUGGET_HIDDEN_STORAGE_DTYPES,
    build_nugget_model,
    load_nugget_checkpoint,
)
from dna_compress.nugget_tokenization import (
    NUGGET_TOKENIZERS,
    apply_nugget_tokenizer_to_model_config,
    build_nugget_tokenizer_spec,
)


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


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.species is not None:
        config.data.species = args.species
    _apply_if_not_none(config, "model.implementation", args.implementation)
    _apply_if_not_none(config, "model.variant", args.variant)
    _apply_if_not_none(config, "model.seq_length", args.seq_length)
    _apply_if_not_none(config, "model.nugget_hidden_mode", args.nugget_hidden_mode)
    _apply_if_not_none(config, "model.nugget_hidden_storage_dtype", args.nugget_hidden_storage_dtype)
    _apply_if_not_none(config, "data.dataset_dir", args.dataset_dir)
    _apply_if_not_none(config, "data.nugget_tokenizer", args.nugget_tokenizer)
    _apply_if_not_none(config, "data.species_prefix_map", args.species_prefix_map)
    _apply_if_not_none(config, "data.train_ratio", args.train_ratio)
    _apply_if_not_none(config, "data.val_ratio", args.val_ratio)
    _apply_if_not_none(config, "data.test_ratio", args.test_ratio)
    _apply_if_not_none(config, "data.max_train_bytes_per_species", args.max_train_bytes)
    _apply_if_not_none(config, "data.max_val_bytes_per_species", args.max_val_bytes)
    _apply_if_not_none(config, "data.max_test_bytes_per_species", args.max_test_bytes)
    _apply_if_not_none(config, "data.compression_sample_bytes", args.compression_sample_bytes)
    _apply_if_not_none(config, "data.token_merge_size", args.token_merge_size)
    _apply_if_not_none(config, "data.token_merge_alphabet", args.token_merge_alphabet)
    _apply_if_not_none(config, "train.device", args.device)
    _apply_if_not_none(config, "train.dtype", args.dtype)
    _apply_if_not_none(config, "train.eval_batch_size", args.eval_batch_size)
    _apply_if_not_none(config, "output.output_dir", args.output_dir)
    _apply_if_not_none(config, "arithmetic.coding_mode", args.arithmetic_coding_mode)
    _apply_if_not_none(config, "arithmetic.merge_size", args.arithmetic_merge_size)
    for item in args.override:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected section.key=value")
        key, raw_value = item.split("=", 1)
        _set_nested_attr(config, key.strip(), _parse_scalar(raw_value.strip()))


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.config is not None:
        return load_experiment_config(args.config)
    if args.run_dir is not None:
        return load_experiment_config(Path(args.run_dir) / "resolved_config.json")
    raise ValueError("Provide --run-dir or --config.")


def _resolve_checkpoint(args: argparse.Namespace, config: ExperimentConfig) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    if args.weight is not None:
        return Path(args.weight)
    if args.run_dir is not None:
        return Path(args.run_dir) / f"{args.checkpoint_tag}.pt"
    return Path(config.output.output_dir) / f"{args.checkpoint_tag}.pt"


def _normalize_splits(raw_splits: list[str]) -> list[str]:
    if "all" in raw_splits:
        return ["train", "val", "test"]
    return raw_splits


def _sources_for_split(splits, split_name: str) -> list[bytes]:
    if split_name == "train":
        return splits.train_sources
    if split_name == "val":
        return splits.val_sources
    if split_name == "test":
        return splits.test_sources
    raise ValueError(f"Unsupported split '{split_name}'")


def _species_names(splits) -> list[str]:
    return [str(item["species"]) for item in splits.summary["species"]]


def _build_fixed_factorizer(config: ExperimentConfig, tokenizer_spec, device: torch.device):
    if config.arithmetic.coding_mode != "fixed_token_units":
        return None
    special_ids = [tokenizer_spec.pad_id]
    if tokenizer_spec.eos_id is not None:
        special_ids.append(tokenizer_spec.eos_id)
    return build_fixed_token_arithmetic_factorizer(
        vocab_size=tokenizer_spec.vocab_size,
        special_token_ids=special_ids,
        model_merge_size=config.data.token_merge_size,
        arithmetic_merge_size=config.arithmetic.merge_size,
        alphabet=tokenizer_spec.token_merge_alphabet,
    ).to(device)


def _build_prefix_trie(config: ExperimentConfig, tokenizer_spec, device: torch.device):
    if config.arithmetic.coding_mode != "base_prefix_exact_gpu_cpu":
        return None
    if tokenizer_spec.tokenizer is None:
        raise ValueError("base_prefix_exact_gpu_cpu requires dnagpt_kmer tokenizer.")
    return build_dnagpt_prefix_trie(tokenizer_spec.tokenizer).to(device)


def _run_split(
    *,
    model,
    config: ExperimentConfig,
    tokenizer_spec,
    split_name: str,
    splits,
    modes: list[str],
    device: torch.device,
    fixed_factorizer,
    prefix_trie,
) -> dict[str, object]:
    sources = _sources_for_split(splits, split_name)
    species_names = _species_names(splits)
    split_result: dict[str, object] = {}
    for mode in modes:
        per_source: list[dict[str, object]] = []
        source_total = len(sources)
        for source_index, (species_name, source) in enumerate(zip(species_names, sources), start=1):
            def _on_progress(batch_done: int, batch_total: int, *, si: int = source_index, sn: str = species_name) -> None:
                ratio = 100.0 * batch_done / max(batch_total, 1)
                print(
                    f"\r[compress] split={split_name} mode={mode} source={si}/{source_total}({sn}) "
                    f"batch={batch_done}/{batch_total} ({ratio:5.1f}%)",
                    end="",
                    flush=True,
                )

            metrics = compress_nugget_source(
                model=model,
                species=species_name,
                source=source,
                tokenizer_spec=tokenizer_spec,
                seq_length=config.model.seq_length,
                device=device,
                dtype_name=config.train.dtype,
                batch_size=config.train.eval_batch_size,
                requested_bytes=config.data.compression_sample_bytes,
                mode=mode,
                arithmetic_frequency_total=config.arithmetic.frequency_total,
                arithmetic_target_uniform_mass=config.arithmetic.target_uniform_mass,
                arithmetic_coding_mode=config.arithmetic.coding_mode,
                arithmetic_merge_size=config.arithmetic.merge_size,
                hidden_mode=config.model.nugget_hidden_mode,
                hidden_storage_dtype=config.model.nugget_hidden_storage_dtype,
                species_prefix_map=config.data.species_prefix_map,
                fixed_factorizer=fixed_factorizer,
                prefix_trie=prefix_trie,
                requires_scores_side_info=not config.model.nugget_straight_through,
                progress_callback=_on_progress,
            )
            print()
            per_source.append({"species": species_name, **metrics})
        split_result[mode] = {
            "aggregate": summarize_nugget_per_source(per_source),
            "per_source": per_source,
        }
    return split_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone Nugget compression on DNACorpus.")
    parser.add_argument("--run-dir")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--weight")
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument("--split", nargs="+", default=["test"], choices=["train", "val", "test", "all"])
    parser.add_argument("--compression-modes", nargs="+", default=[NON_OVERLAP_MODE], choices=list(SUPPORTED_NUGGET_COMPRESSION_MODES))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--implementation", choices=["nugget"])
    parser.add_argument("--variant", choices=["dna_gpt0.1b_h", "dna_gpt0.1b_m", "dna_gpt3b_m"])
    parser.add_argument("--seq-length", type=int)
    parser.add_argument("--nugget-hidden-mode", choices=list(NUGGET_HIDDEN_MODES))
    parser.add_argument("--nugget-hidden-storage-dtype", choices=list(NUGGET_HIDDEN_STORAGE_DTYPES))
    parser.add_argument("--dataset-dir")
    parser.add_argument("--species", nargs="+")
    parser.add_argument("--nugget-tokenizer", choices=list(NUGGET_TOKENIZERS))
    parser.add_argument("--species-prefix-map", type=json.loads)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    parser.add_argument("--max-train-bytes", type=int)
    parser.add_argument("--max-val-bytes", type=int)
    parser.add_argument("--max-test-bytes", type=int)
    parser.add_argument("--compression-sample-bytes", type=int)
    parser.add_argument("--token-merge-size", type=int)
    parser.add_argument("--token-merge-alphabet")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--arithmetic-coding-mode", choices=list(NUGGET_ARITHMETIC_CODING_MODES))
    parser.add_argument("--arithmetic-merge-size", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _resolve_config(args)
    _apply_overrides(config, args)
    tokenizer_spec = build_nugget_tokenizer_spec(config.data, config.model)
    apply_nugget_tokenizer_to_model_config(config.model, tokenizer_spec)
    validate_nugget_config(config)
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))

    device = resolve_device(config.train.device)
    model, backbone_spec = build_nugget_model(config.model, tokenizer_spec)
    checkpoint_path = _resolve_checkpoint(args, config)
    model_state, checkpoint_metadata, _ = load_nugget_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    fixed_factorizer = _build_fixed_factorizer(config, tokenizer_spec, device)
    prefix_trie = _build_prefix_trie(config, tokenizer_spec, device)
    splits = load_splits(config.data, seq_length=config.model.seq_length)
    requested_splits = _normalize_splits(args.split)
    metrics: dict[str, object] = {
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_metadata.get("step"),
        "best_val_bpb": checkpoint_metadata.get("best_val_bpb"),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "resolved_config": config.to_dict(),
        "dataset": splits.summary,
        "nugget": {
            "backbone": backbone_spec.backbone,
            "tokenizer": tokenizer_spec.name,
            "vocab_size": tokenizer_spec.vocab_size,
            "pad_id": tokenizer_spec.pad_id,
            "eos_id": tokenizer_spec.eos_id,
            "decoder_start_token_id": backbone_spec.decoder_start_token_id,
            "decoder_start_source": backbone_spec.decoder_start_source,
            "config_source": backbone_spec.config_source,
            "d_model": backbone_spec.d_model,
            "encoder_layers": backbone_spec.encoder_layers,
            "decoder_layers": backbone_spec.decoder_layers,
            "encoder_attention_heads": backbone_spec.encoder_attention_heads,
            "decoder_attention_heads": backbone_spec.decoder_attention_heads,
            "encoder_ffn_dim": backbone_spec.encoder_ffn_dim,
            "decoder_ffn_dim": backbone_spec.decoder_ffn_dim,
            "seq_length": config.model.seq_length,
            "nugget_ratio": config.model.nugget_ratio,
            "hidden_mode": config.model.nugget_hidden_mode,
            "hidden_storage_dtype": config.model.nugget_hidden_storage_dtype,
            "arithmetic_coding_mode": config.arithmetic.coding_mode,
            "arithmetic_merge_size": config.arithmetic.merge_size,
        },
        "results": {},
    }
    for split_name in requested_splits:
        print(f"[compress] split={split_name} modes={','.join(args.compression_modes)}")
        metrics["results"][split_name] = _run_split(
            model=model,
            config=config,
            tokenizer_spec=tokenizer_spec,
            split_name=split_name,
            splits=splits,
            modes=args.compression_modes,
            device=device,
            fixed_factorizer=fixed_factorizer,
            prefix_trie=prefix_trie,
        )

    if args.output_json:
        output_json = Path(args.output_json)
    elif args.run_dir is not None:
        output_json = Path(args.run_dir) / "nugget_compression_compare.json"
    else:
        output_json = Path(config.output.output_dir) / "nugget_compression_compare.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved Nugget compression metrics to {output_json}")


if __name__ == "__main__":
    main()
