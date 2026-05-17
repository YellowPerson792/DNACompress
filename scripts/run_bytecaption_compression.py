from __future__ import annotations

"""Run standalone ByteCaption/PureT compression on DNACorpus."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.bytecaption_compression import (
    BYTECAPTION_ARITHMETIC_CODING_MODES,
    SUPPORTED_BYTECAPTION_COMPRESSION_MODES,
    compress_bytecaption_source,
    summarize_bytecaption_per_source,
)
from dna_compress.bytecaption_experiment import validate_bytecaption_config
from dna_compress.bytecaption_loader import BYTECAPTION_HIDDEN_STORAGE_DTYPES, build_bytecaption_model, load_bytecaption_checkpoint
from dna_compress.bytecaption_tokenization import (
    BYTECAPTION_TOKENIZERS,
    apply_bytecaption_tokenizer_to_model_config,
    build_bytecaption_tokenizer_spec,
)
from dna_compress.config import ExperimentConfig, load_experiment_config
from dna_compress.data import load_splits
from dna_compress.experiment import resolve_device
from dna_compress.fixed_token_factorization import build_fixed_token_arithmetic_factorizer


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
    _apply_if_not_none(config, "model.seq_length", args.seq_length)
    _apply_if_not_none(config, "model.bytecaption_latent_mode", args.bytecaption_latent_mode)
    _apply_if_not_none(config, "model.bytecaption_code_dim", args.bytecaption_code_dim)
    _apply_if_not_none(config, "model.bytecaption_flatten_bottleneck_dim", args.bytecaption_flatten_bottleneck_dim)
    _apply_if_not_none(config, "model.bytecaption_hidden_storage_dtype", args.bytecaption_hidden_storage_dtype)
    _apply_if_not_none(config, "data.dataset_dir", args.dataset_dir)
    _apply_if_not_none(config, "data.nugget_tokenizer", args.bytecaption_tokenizer)
    _apply_if_not_none(config, "data.token_merge_size", args.token_merge_size)
    _apply_if_not_none(config, "data.token_merge_alphabet", args.token_merge_alphabet)
    _apply_if_not_none(config, "data.compression_sample_bytes", args.compression_sample_bytes)
    _apply_if_not_none(config, "train.device", args.device)
    _apply_if_not_none(config, "train.dtype", args.dtype)
    _apply_if_not_none(config, "train.eval_batch_size", args.eval_batch_size)
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
    if args.run_dir is not None:
        return Path(args.run_dir) / f"{args.checkpoint_tag}.pt"
    return Path(config.output.output_dir) / f"{args.checkpoint_tag}.pt"


def _sources_for_split(splits, split: str) -> list[bytes]:
    if split == "train":
        return splits.train_sources
    if split == "val":
        return splits.val_sources
    if split == "test":
        return splits.test_sources
    raise ValueError(f"Unsupported split: {split}")


def _species_names(splits) -> list[str]:
    return [str(item["species"]) for item in splits.summary["species"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-tag", default="best")
    parser.add_argument("--split", nargs="+", default=["test"], choices=["train", "val", "test", "all"])
    parser.add_argument("--compression-modes", nargs="+", default=["windows_nonoverlap"], choices=SUPPORTED_BYTECAPTION_COMPRESSION_MODES)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--implementation", default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--bytecaption-latent-mode", default=None)
    parser.add_argument("--bytecaption-code-dim", type=int, default=None)
    parser.add_argument("--bytecaption-flatten-bottleneck-dim", type=int, default=None)
    parser.add_argument("--bytecaption-hidden-storage-dtype", choices=BYTECAPTION_HIDDEN_STORAGE_DTYPES, default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--species", nargs="+", default=None)
    parser.add_argument("--bytecaption-tokenizer", choices=BYTECAPTION_TOKENIZERS, default=None)
    parser.add_argument("--token-merge-size", type=int, default=None)
    parser.add_argument("--token-merge-alphabet", default=None)
    parser.add_argument("--compression-sample-bytes", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--arithmetic-coding-mode", choices=BYTECAPTION_ARITHMETIC_CODING_MODES, default=None)
    parser.add_argument("--arithmetic-merge-size", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if "all" in args.split:
        args.split = ["train", "val", "test"]
    config = _resolve_config(args)
    _apply_overrides(config, args)
    tokenizer_spec = build_bytecaption_tokenizer_spec(config.data, config.model)
    apply_bytecaption_tokenizer_to_model_config(config.model, tokenizer_spec)
    validate_bytecaption_config(config)
    checkpoint = _resolve_checkpoint(args, config)
    device = resolve_device(config.train.device)
    model = build_bytecaption_model(config.model).to(device)
    load_bytecaption_checkpoint(model, checkpoint, strict=True)
    model.eval()

    splits = load_splits(config.data)
    species = _species_names(splits)
    fixed_factorizer = None
    if config.arithmetic.coding_mode == "fixed_token_units":
        fixed_factorizer = build_fixed_token_arithmetic_factorizer(
            token_merge_size=config.data.token_merge_size,
            token_merge_alphabet=config.data.token_merge_alphabet,
            pad_id=tokenizer_spec.pad_id,
            eos_id=tokenizer_spec.eos_id,
        )

    results: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "config": config.to_dict(),
        "splits": {},
    }
    for split_name in args.split:
        per_source = []
        for index, source in enumerate(_sources_for_split(splits, split_name)):
            species_name = species[index] if index < len(species) else f"source_{index}"
            for mode in args.compression_modes:
                print(f"[bytecaption-compress] split={split_name} species={species_name} mode={mode}")
                row = compress_bytecaption_source(
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
                    hidden_storage_dtype=config.model.bytecaption_hidden_storage_dtype,
                    species_prefix_map=config.data.species_prefix_map,
                    fixed_factorizer=fixed_factorizer,
                )
                row["species"] = species_name
                per_source.append(row)
        summary = summarize_bytecaption_per_source(per_source)
        results["splits"][split_name] = {"per_source": per_source, "summary": summary}
        print(
            f"[bytecaption-compress] split={split_name} "
            f"decoder_bpb={summary['total_decoder_theoretical_bits_per_base']:.4f} "
            f"theoretical_bpb={summary['total_theoretical_bits_per_base']:.4f} "
            f"total_bpb={summary['total_bits_per_base']:.4f}"
        )

    output_json = Path(args.output_json) if args.output_json else Path(config.output.output_dir) / "bytecaption_compression_compare.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[bytecaption-compress] wrote {output_json}")


if __name__ == "__main__":
    main()
