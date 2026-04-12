from __future__ import annotations

"""Compare Megabyte compression procedures on selected DNA splits.

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

Modes:
  - sliding_token: original per-token right-aligned sliding window evaluation
  - train_windows_nonoverlap: contiguous non-overlapping windows, matching training-style inputs
  - train_windows_overlap: contiguous overlapping windows with patch-aligned stride;
    exact cache reuse is not enabled in this evaluator

Complete examples:

    python scripts/run_dna_compression.py \
      --run-dir outputs/dna_megabyte_large_conv_all \
      --checkpoint-tag best \
      --dataset-dir datasets/ensembl_raw \
      --sequence-source-mode fasta_dir \
      --multi-sequence-mode separate \
      --split train val test \
      --eval-batch-size 10 \
      --compression-modes train_windows_nonoverlap \
      --compression-sample-bytes 60000 \
      --species homo_sapiens mus_musculus bos_taurus danio_rerio \
                drosophila_melanogaster caenorhabditis_elegans \
                saccharomyces_cerevisiae arabidopsis_thaliana \
      --arithmetic-coding-mode base_prefix_exact_gpu_cpu \
      --arithmetic-merge-size 3
      
      --device cuda:2 
      --parallel-window-arithmetic \
      --arithmetic-workers 0

    python scripts/run_dna_compression.py \
      --run-dir outputs\\dna_megabyte_quick_l1024_p3 \
      --dataset-dir datasets/ensembl_raw \
      --sequence-source-mode fasta_dir \
      --multi-sequence-mode separate \
      --split train val test \
      --compression-modes train_windows_overlap \
      --overlap-patches 128 \
      --species homo_sapiens mus_musculus bos_taurus danio_rerio \
                drosophila_melanogaster caenorhabditis_elegans \
                saccharomyces_cerevisiae arabidopsis_thaliana \
      --output-json outputs/dna_megabyte_quick_l1024_p3/compression_compare.json \
      --export-out-dir outputs/dna_megabyte_quick_l1024_p3/statistics

Compatibility (explicit paths still supported):

        python scripts/run_dna_compression.py \
            --config outputs\\dna_megabyte_quick_l1024_p3\\resolved_config.json \
            --checkpoint outputs\\dna_megabyte_quick_l1024_p3\\best.pt \
            --split test
"""
 
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress import load_experiment_config
from dna_compress.compression_eval import (
    MEGABYTE_ARITHMETIC_CODING_MODES,
    NON_OVERLAP_MODE,
    OVERLAP_MODE,
    SLIDING_TOKEN_MODE,
    SUPPORTED_COMPRESSION_MODES,
    compress_source,
    resolve_device,
    summarize_per_source,
)
from dna_compress.config import ExperimentConfig
from dna_compress.data import load_splits
from dna_compress.fixed_token_factorization import build_fixed_token_arithmetic_factorizer
from dna_compress.megabyte_loader import build_model
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


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.species is not None:
        config.data.species = args.species

    _apply_if_not_none(config, "model.implementation", args.implementation)
    _apply_if_not_none(config, "model.seq_length", args.seq_length)
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
    _apply_if_not_none(config, "data.token_merge_size", args.token_merge_size)
    _apply_if_not_none(config, "data.token_merge_alphabet", args.token_merge_alphabet)
    _apply_if_not_none(config, "data.compression_sample_bytes", args.compression_sample_bytes)
    _apply_if_not_none(config, "arithmetic.coding_mode", args.arithmetic_coding_mode)
    _apply_if_not_none(config, "arithmetic.merge_size", args.arithmetic_merge_size)
    _apply_if_not_none(config, "train.device", args.device)
    _apply_if_not_none(config, "train.dtype", args.dtype)
    _apply_if_not_none(config, "train.eval_batch_size", args.eval_batch_size)
    _apply_if_not_none(config, "output.output_dir", args.output_dir)

    for item in args.override:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected section.key=value")
        key, raw_value = item.split("=", 1)
        _set_nested_attr(config, key.strip(), _parse_scalar(raw_value.strip()))


def _normalize_splits(raw_splits: list[str]) -> list[str]:
    if "all" in raw_splits:
        return ["train", "val", "test"]
    return raw_splits


def _resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return Path(args.config)
    if args.run_dir is not None:
        return Path(args.run_dir) / "resolved_config.json"
    raise ValueError("Either --run-dir or --config must be provided")


def _checkpoint_path(args: argparse.Namespace, config: ExperimentConfig) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    if args.run_dir is not None:
        return Path(args.run_dir) / f"{args.checkpoint_tag}.pt"
    return Path(config.output.output_dir) / f"{args.checkpoint_tag}.pt"


def _load_model(config: ExperimentConfig, checkpoint_path: Path, device: torch.device):
    model = build_model(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint '{checkpoint_path}' is missing 'model_state'")
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


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


def _source_entries(splits) -> list[dict[str, object]]:
    return [dict(item) for item in splits.summary["species"]]


def _validate_args(config: ExperimentConfig, args: argparse.Namespace) -> None:
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
    if config.model.input_causal_conv_kernel_size <= 0:
        raise ValueError("model.input_causal_conv_kernel_size must be >= 1")
    if args.overlap_patches is not None and args.overlap_patches <= 0:
        raise ValueError("--overlap-patches must be > 0")
    overlap_stride = _resolve_overlap_stride(config, args)
    if overlap_stride <= 0:
        raise ValueError("overlap stride must be > 0")
    if OVERLAP_MODE in args.compression_modes and overlap_stride >= config.model.seq_length:
        raise ValueError("overlap stride must be smaller than model.seq_length for overlap mode")
    if OVERLAP_MODE in args.compression_modes and overlap_stride % config.model.patch_size != 0:
        raise ValueError("overlap stride must be a multiple of model.patch_size")
    if config.data.token_merge_size <= 0:
        raise ValueError("data.token_merge_size must be >= 1")
    normalize_alphabet(config.data.token_merge_alphabet)
    if config.data.sequence_source_mode not in {"auto", "flat_file", "fasta_dir"}:
        raise ValueError("data.sequence_source_mode must be one of: auto, flat_file, fasta_dir")
    if config.data.multi_sequence_mode not in {"separate", "concat"}:
        raise ValueError("data.multi_sequence_mode must be one of: separate, concat")
    if not isinstance(config.data.sequence_include_map, dict):
        raise ValueError("data.sequence_include_map must be a dict[str, list[str]]")
    if config.arithmetic.frequency_total is not None and config.arithmetic.frequency_total <= 0:
        raise ValueError("arithmetic.frequency_total must be > 0 when provided")
    if not (0.0 < config.arithmetic.target_uniform_mass <= 1.0):
        raise ValueError("arithmetic.target_uniform_mass must be in (0.0, 1.0]")
    if config.arithmetic.coding_mode not in MEGABYTE_ARITHMETIC_CODING_MODES:
        raise ValueError(
            "arithmetic.coding_mode must be one of: "
            + ", ".join(MEGABYTE_ARITHMETIC_CODING_MODES)
        )
    if config.arithmetic.coding_mode == "base_prefix_exact_gpu_cpu":
        if config.arithmetic.merge_size <= 0:
            raise ValueError("arithmetic.merge_size must be >= 1")
        if config.arithmetic.merge_size > config.data.token_merge_size:
            raise ValueError("arithmetic.merge_size must be <= data.token_merge_size")


def _resolve_overlap_stride(config: ExperimentConfig, args: argparse.Namespace) -> int:
    if args.overlap_patches is not None:
        return args.overlap_patches * config.model.patch_size
    if args.overlap_stride is None:
        return config.model.patch_size
    return args.overlap_stride


def _run_split(
    *,
    model: torch.nn.Module,
    config: ExperimentConfig,
    split_name: str,
    splits,
    modes: list[str],
    overlap_stride: int,
    device: torch.device,
    factorizer,
) -> dict[str, object]:
    sources = _sources_for_split(splits, split_name)
    source_entries = _source_entries(splits)
    split_result: dict[str, object] = {}

    for mode in modes:
        per_source: list[dict[str, object]] = []
        source_total = len(sources)
        for source_index, (entry, source) in enumerate(zip(source_entries, sources), start=1):
            species_name = str(entry["species"])
            source_name = str(entry.get("source_name", species_name))

            def _on_progress(batch_done: int, batch_total: int, *, si: int = source_index, sn: str = source_name) -> None:
                ratio = 100.0 * batch_done / max(batch_total, 1)
                print(
                    (
                        f"\r[compress] split={split_name} mode={mode} "
                        f"source={si}/{source_total}({sn}) batch={batch_done}/{batch_total} ({ratio:5.1f}%)"
                    ),
                    end="",
                    flush=True,
                )

            metrics = compress_source(
                model=model,
                source=source,
                seq_length=config.model.seq_length,
                pad_id=config.model.pad_id,
                eos_id=config.model.eos_id,
                device=device,
                dtype_name=config.train.dtype,
                batch_size=config.train.eval_batch_size,
                requested_bytes=config.data.compression_sample_bytes,
                mode=mode,
                overlap_stride=overlap_stride,
                token_merge_size=config.data.token_merge_size,
                token_merge_alphabet=config.data.token_merge_alphabet,
                arithmetic_frequency_total=config.arithmetic.frequency_total,
                arithmetic_target_uniform_mass=config.arithmetic.target_uniform_mass,
                arithmetic_coding_mode=config.arithmetic.coding_mode,
                arithmetic_merge_size=config.arithmetic.merge_size,
                factorizer=factorizer,
                progress_callback=_on_progress,
            )
            print()
            per_source.append({"species": species_name, "source_name": source_name, **metrics})

        split_result[mode] = {
            "aggregate": summarize_per_source(per_source),
            "per_source": per_source,
        }

    return split_result


def _run_local_payload_export(
    *,
    run_dir: Path,
    compression_json_name: str,
    export_out_dir: str | None,
    export_project: str,
    export_entity: str,
    export_name: str | None,
) -> None:
    export_script = REPO_ROOT / "scripts" / "export_statistics.py"
    if not export_script.exists():
        print(f"[export] skip: script not found: {export_script}")
        return

    command = [
        sys.executable,
        str(export_script),
        "--run-dir",
        str(run_dir),
        "--compression-json",
        compression_json_name,
    ]

    if export_out_dir:
        command.extend(["--out-dir", export_out_dir])
    if export_project:
        command.extend(["--project", export_project])
    if export_entity:
        command.extend(["--entity", export_entity])
    if export_name:
        command.extend(["--name", export_name])

    print(f"[export] running local payload export for run_dir={run_dir}")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0:
        if completed.stderr.strip():
            print(completed.stderr.strip())
        print(f"[export] warning: export script failed with exit code {completed.returncode}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone DNA compression comparisons with a trained Megabyte checkpoint.",
    )
    parser.add_argument(
        "--run-dir",
        help="Preferred input: experiment output directory containing resolved_config.json and best/last checkpoints.",
    )
    parser.add_argument("--config", help="Path to experiment JSON config. Optional when --run-dir is provided.")
    parser.add_argument("--checkpoint", help="Explicit checkpoint path. Defaults to output_dir/<checkpoint-tag>.pt")
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument(
        "--split",
        nargs="+",
        default=["test"],
        choices=["train", "val", "test", "all"],
        help="Which data splits to compress.",
    )
    parser.add_argument(
        "--compression-modes",
        nargs="+",
        default=list(SUPPORTED_COMPRESSION_MODES),
        choices=list(SUPPORTED_COMPRESSION_MODES),
        help="Compression procedures to run.",
    )
    parser.add_argument(
        "--overlap-stride",
        type=int,
        default=None,
        help="Overlap stride in tokens for train_windows_overlap mode. Must be a multiple of patch_size.",
    )
    parser.add_argument(
        "--overlap-patches",
        type=int,
        default=None,
        help="Preferred overlap stride measured in patches. Effective stride = overlap_patches * patch_size.",
    )
    parser.add_argument("--print-config", action="store_true", help="Print resolved config before running.")
    parser.add_argument("--output-json", help="Where to save JSON metrics. Defaults to output_dir/compression_compare.json")
    parser.add_argument(
        "--no-auto-export",
        action="store_true",
        help="Disable automatic execution of scripts/export_statistics.py after compression finishes.",
    )
    parser.add_argument(
        "--export-out-dir",
        default=None,
        help="Optional output directory for exported tables. Defaults to <run-dir>/statistics.",
    )
    parser.add_argument("--export-project", default="", help="Optional project metadata for exported run_metadata.json.")
    parser.add_argument("--export-entity", default="", help="Optional entity metadata for exported run_metadata.json.")
    parser.add_argument("--export-name", default=None, help="Optional run name for exported run_metadata.json.")
    parser.add_argument("--override", action="append", default=[], help="Generic override in form section.key=value.")

    model_group = parser.add_argument_group("model/data overrides")
    model_group.add_argument(
        "--implementation",
        choices=["megabyte", "megabyte_in_action", "megabyte_in_action_causal_conv", "megabyte_relative"],
    )
    model_group.add_argument("--seq-length", type=int)
    model_group.add_argument("--input-causal-conv-kernel-size", type=int)
    model_group.add_argument("--dataset-dir")
    model_group.add_argument("--species", nargs="+")
    model_group.add_argument("--sequence-source-mode", choices=["auto", "flat_file", "fasta_dir"])
    model_group.add_argument("--multi-sequence-mode", choices=["separate", "concat"])
    model_group.add_argument("--clean-cache-enabled", dest="clean_cache_enabled", action="store_true", default=None)
    model_group.add_argument("--no-clean-cache", dest="clean_cache_enabled", action="store_false")
    model_group.add_argument("--clean-cache-dir")
    model_group.add_argument(
        "--sequence-include",
        action="append",
        help="Repeatable sequence selector in form species=key1,key2,...",
    )
    model_group.add_argument("--train-ratio", type=float)
    model_group.add_argument("--val-ratio", type=float)
    model_group.add_argument("--test-ratio", type=float)
    model_group.add_argument("--max-train-bytes", type=int)
    model_group.add_argument("--max-val-bytes", type=int)
    model_group.add_argument("--max-test-bytes", type=int)
    model_group.add_argument("--token-merge-size", type=int)
    model_group.add_argument("--token-merge-alphabet")
    model_group.add_argument("--compression-sample-bytes", type=int)

    runtime_group = parser.add_argument_group("runtime overrides")
    runtime_group.add_argument("--device", help="auto/cpu/cuda/cuda:0 ...")
    runtime_group.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    runtime_group.add_argument("--eval-batch-size", type=int)
    runtime_group.add_argument("--output-dir")
    runtime_group.add_argument(
        "--arithmetic-coding-mode",
        choices=list(MEGABYTE_ARITHMETIC_CODING_MODES),
        help="Arithmetic coding path for Megabyte compression.",
    )
    runtime_group.add_argument("--arithmetic-merge-size", type=int)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config_path = _resolve_config_path(args)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_experiment_config(config_path)
    _apply_overrides(config, args)
    _validate_args(config, args)
    apply_token_merge_to_model_config(config.model, config.data)

    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))

    device = resolve_device(config.train.device)
    checkpoint_path = _checkpoint_path(args, config)
    model, checkpoint = _load_model(config, checkpoint_path, device)
    factorizer = None
    if config.arithmetic.coding_mode == "base_prefix_exact_gpu_cpu":
        factorizer = build_fixed_token_arithmetic_factorizer(
            vocab_size=config.model.vocab_size,
            special_token_ids=[config.model.pad_id, config.model.eos_id],
            model_merge_size=config.data.token_merge_size,
            arithmetic_merge_size=config.arithmetic.merge_size,
            alphabet=normalize_alphabet(config.data.token_merge_alphabet),
        ).to(device)
    print("[compress] loading data splits...", flush=True)
    split_started = time.time()
    splits = load_splits(config.data, seq_length=config.model.seq_length)
    split_entries = splits.summary["species"]
    clean_cache_summary = splits.summary.get("clean_cache", {})
    print(
        f"[compress] data splits loaded: sources={len(split_entries)} elapsed={time.time() - split_started:.1f}s",
        flush=True,
    )
    if isinstance(clean_cache_summary, dict) and int(clean_cache_summary.get("applicable_sources", 0)) > 0:
        print(
            "[cache] clean "
            f"enabled={bool(clean_cache_summary.get('enabled'))} "
            f"dir={clean_cache_summary.get('cache_dir')} "
            f"hits={int(clean_cache_summary.get('hits', 0))} "
            f"created={int(clean_cache_summary.get('created', 0))} "
            f"rebuilt={int(clean_cache_summary.get('rebuilt', 0))} "
            f"disabled={int(clean_cache_summary.get('disabled', 0))}",
            flush=True,
        )
    requested_splits = _normalize_splits(args.split)
    overlap_stride = _resolve_overlap_stride(config, args)
    metrics = {
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "best_val_bpb": checkpoint.get("best_val_bpb"),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "overlap_stride_tokens": overlap_stride,
        "overlap_stride_patches": overlap_stride // config.model.patch_size,
        "resolved_config": config.to_dict(),
        "dataset": splits.summary,
        "results": {},
    }

    for split_name in requested_splits:
        print(f"[compress] split={split_name} modes={','.join(args.compression_modes)}")
        metrics["results"][split_name] = _run_split(
            model=model,
            config=config,
            split_name=split_name,
            splits=splits,
            modes=args.compression_modes,
            overlap_stride=overlap_stride,
            device=device,
            factorizer=factorizer,
        )
        if "arithmetic" not in metrics:
            split_result = metrics["results"][split_name]
            if isinstance(split_result, dict):
                for mode_payload in split_result.values():
                    if not isinstance(mode_payload, dict):
                        continue
                    aggregate = mode_payload.get("aggregate")
                    if not isinstance(aggregate, dict):
                        continue
                    if "arithmetic_frequency_total" in aggregate:
                        metrics["arithmetic"] = {
                            "frequency_total": aggregate.get("arithmetic_frequency_total"),
                            "vocab_size": aggregate.get("arithmetic_vocab_size"),
                            "target_uniform_mass": aggregate.get("arithmetic_target_uniform_mass"),
                            "effective_uniform_mass": aggregate.get("arithmetic_effective_uniform_mass"),
                            "coding_mode": aggregate.get("arithmetic_coding_mode"),
                            "merge_size": aggregate.get("arithmetic_merge_size"),
                        }
                        break

    if args.output_json:
        output_json = Path(args.output_json)
    elif args.run_dir is not None:
        output_json = Path(args.run_dir) / "compression_compare.json"
    else:
        output_json = Path(config.output.output_dir) / "compression_compare.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved compression metrics to {output_json}")

    if not args.no_auto_export:
        export_run_dir = Path(args.run_dir) if args.run_dir is not None else output_json.parent
        _run_local_payload_export(
            run_dir=export_run_dir,
            compression_json_name=output_json.name,
            export_out_dir=args.export_out_dir,
            export_project=args.export_project,
            export_entity=args.export_entity,
            export_name=args.export_name,
        )


if __name__ == "__main__":
    main()
