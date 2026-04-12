from __future__ import annotations

"""Run standalone DNAGPT compression on DNACorpus.

Example: evaluate official DNAGPT 0.1B multi-organism weights on HoSa test split
        
    python scripts/run_dnagpt_compression.py \
      --split train val test \
      --eval-batch-size 10 \
      --config configs/dna_dnagpt_h_quick.json \
      --weight third_party/DNAGPT/checkpoints/dna_gpt0.1b_h.pth \
      --compression-modes train_windows_nonoverlap \
      --compression-sample-bytes 60000 \
      --train-ratio 0.6 \
      --val-ratio 0.2 \
      --test-ratio 0.2 \
      --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
      --arithmetic-coding-mode base_prefix_exact_gpu_cpu \
      --arithmetic-merge-size 3 \
      --output-dir outputs/dna_dnagpt_0p1bh_all \
      --output-json outputs/dna_dnagpt_0p1bh_all/compression_compare.json \
      --export-out-dir outputs/dna_dnagpt_0p1bh_all/statistics 
      
      --device cuda:2 \
      
    python scripts/run_dnagpt_compression.py \
      --split train val test \
      --eval-batch-size 10 \
      --run-dir outputs/dna_dnagpt_0p1bm_all_finetune \
      --compression-modes train_windows_nonoverlap \
      --compression-sample-bytes 60000 \
      --arithmetic-coding-mode base_prefix_exact_gpu_cpu \
      --arithmetic-merge-size 2 \
      --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe 

"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.compression_eval import NON_OVERLAP_MODE, SLIDING_TOKEN_MODE, summarize_per_source
from dna_compress.config import ExperimentConfig, load_experiment_config
from dna_compress.data import load_splits
from dna_compress.dnagpt_compression import (
    DNAGPT_ARITHMETIC_CODING_MODES,
    SUPPORTED_DNAGPT_COMPRESSION_MODES,
    compress_dnagpt_source,
)
from dna_compress.dnagpt_experiment import validate_dnagpt_config
from dna_compress.dnagpt_loader import (
    build_dnagpt_components,
    default_pretrained_weight_path,
    get_variant_spec,
    load_dnagpt_checkpoint,
)
from dna_compress.dnagpt_prefix_coding import build_dnagpt_prefix_trie
from dna_compress.experiment import resolve_device


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


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.species is not None:
        config.data.species = args.species

    _apply_if_not_none(config, "model.implementation", args.implementation)
    _apply_if_not_none(config, "model.variant", args.variant)
    _apply_if_not_none(config, "model.pretrained_weight_path", args.weight)
    _apply_if_not_none(config, "model.seq_length", args.seq_length)

    _apply_if_not_none(config, "data.dataset_dir", args.dataset_dir)
    _apply_if_not_none(config, "data.species_prefix_map", args.species_prefix_map)
    _apply_if_not_none(config, "data.train_ratio", args.train_ratio)
    _apply_if_not_none(config, "data.val_ratio", args.val_ratio)
    _apply_if_not_none(config, "data.test_ratio", args.test_ratio)
    _apply_if_not_none(config, "data.max_train_bytes_per_species", args.max_train_bytes)
    _apply_if_not_none(config, "data.max_val_bytes_per_species", args.max_val_bytes)
    _apply_if_not_none(config, "data.max_test_bytes_per_species", args.max_test_bytes)
    _apply_if_not_none(config, "data.compression_sample_bytes", args.compression_sample_bytes)

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


def _normalize_splits(raw_splits: list[str]) -> list[str]:
    if "all" in raw_splits:
        return ["train", "val", "test"]
    return raw_splits


def _build_default_config(variant: str) -> ExperimentConfig:
    spec = get_variant_spec(variant)
    config = ExperimentConfig()
    config.model.implementation = "dnagpt"
    config.model.variant = variant
    config.model.seq_length = spec.max_len
    config.train.dtype = "float16"
    config.data.token_merge_size = 1
    config.output.run_name = f"dnagpt_{variant}_compression"
    config.output.output_dir = f"outputs/dnagpt_{variant}_compression"
    return config


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.config is not None:
        return load_experiment_config(args.config)
    if args.run_dir is not None:
        run_config = Path(args.run_dir) / "resolved_config.json"
        if run_config.exists():
            return load_experiment_config(run_config)
    if args.variant is None:
        raise ValueError("Provide --config/--run-dir or explicitly set --variant for standalone compression.")
    return _build_default_config(args.variant)


def _resolve_checkpoint_path(args: argparse.Namespace, config: ExperimentConfig) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    if args.run_dir is not None:
        candidate = Path(args.run_dir) / f"{args.checkpoint_tag}.pt"
        if candidate.exists():
            return candidate
    if config.model.pretrained_weight_path:
        return Path(config.model.pretrained_weight_path)
    return default_pretrained_weight_path(config.model.variant)


def _species_names(splits) -> list[str]:
    return [str(item["species"]) for item in splits.summary["species"]]


def _run_split(
    *,
    model: torch.nn.Module,
    tokenizer,
    prefix_trie,
    config: ExperimentConfig,
    spec,
    split_name: str,
    splits,
    modes: list[str],
    device: torch.device,
) -> dict[str, object]:
    if split_name == "train":
        sources = splits.train_sources
    elif split_name == "val":
        sources = splits.val_sources
    elif split_name == "test":
        sources = splits.test_sources
    else:
        raise ValueError(f"Unsupported split '{split_name}'")

    species_names = _species_names(splits)
    split_result: dict[str, object] = {}

    for mode in modes:
        per_source: list[dict[str, object]] = []
        source_total = len(sources)
        for source_index, (species_name, source) in enumerate(zip(species_names, sources), start=1):
            def _on_progress(batch_done: int, batch_total: int, *, si: int = source_index, sn: str = species_name) -> None:
                ratio = 100.0 * batch_done / max(batch_total, 1)
                print(
                    (
                        f"\r[compress] split={split_name} mode={mode} "
                        f"source={si}/{source_total}({sn}) batch={batch_done}/{batch_total} ({ratio:5.1f}%)"
                    ),
                    end="",
                    flush=True,
                )

            metrics = compress_dnagpt_source(
                model=model,
                species=species_name,
                source=source,
                tokenizer=tokenizer,
                kmer_size=spec.kmer_size,
                dynamic_kmer=spec.dynamic_kmer,
                species_prefix_map=config.data.species_prefix_map,
                seq_length=config.model.seq_length,
                pad_id=tokenizer.pad_id,
                device=device,
                dtype_name=config.train.dtype,
                batch_size=config.train.eval_batch_size,
                requested_bytes=config.data.compression_sample_bytes,
                mode=mode,
                arithmetic_frequency_total=config.arithmetic.frequency_total,
                arithmetic_target_uniform_mass=config.arithmetic.target_uniform_mass,
                arithmetic_coding_mode=config.arithmetic.coding_mode,
                arithmetic_merge_size=config.arithmetic.merge_size,
                prefix_trie=prefix_trie,
                progress_callback=_on_progress,
            )
            print()
            per_source.append({"species": species_name, **metrics})

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
    export_entity: str,
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
    if export_entity:
        command.extend(["--entity", export_entity])

    print(f"[export] running local payload export for run_dir={run_dir}")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0:
        if completed.stderr.strip():
            print(completed.stderr.strip())
        print(f"[export] warning: export script failed with exit code {completed.returncode}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone DNAGPT compression on DNACorpus.")
    parser.add_argument("--run-dir", help="Experiment output directory containing resolved_config.json and checkpoints.")
    parser.add_argument("--config", help="Path to DNAGPT experiment JSON config.")
    parser.add_argument("--variant", choices=["dna_gpt0.1b_h", "dna_gpt0.1b_m", "dna_gpt3b_m"])
    parser.add_argument("--weight", help="Official or explicit checkpoint path.")
    parser.add_argument("--checkpoint", help="Explicit checkpoint path. Defaults to run_dir/<checkpoint-tag>.pt or --weight.")
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument("--split", nargs="+", default=["test"], choices=["train", "val", "test", "all"])
    parser.add_argument(
        "--compression-modes",
        nargs="+",
        default=list(SUPPORTED_DNAGPT_COMPRESSION_MODES),
        choices=list(SUPPORTED_DNAGPT_COMPRESSION_MODES),
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--output-json", help="Where to save JSON metrics. Defaults to run_dir/compression_compare.json")
    parser.add_argument("--no-auto-export", action="store_true")
    parser.add_argument("--export-out-dir", default=None)
    parser.add_argument("--export-entity", default="")
    parser.add_argument("--override", action="append", default=[], help="Generic override in form section.key=value.")

    model_group = parser.add_argument_group("model/data overrides")
    model_group.add_argument("--implementation", choices=["dnagpt"])
    model_group.add_argument("--seq-length", type=int)
    model_group.add_argument("--dataset-dir")
    model_group.add_argument("--species", nargs="+")
    model_group.add_argument("--species-prefix-map", type=json.loads, help='JSON dict, e.g. {"OrSa":"R"}')
    model_group.add_argument("--train-ratio", type=float)
    model_group.add_argument("--val-ratio", type=float)
    model_group.add_argument("--test-ratio", type=float)
    model_group.add_argument("--max-train-bytes", type=int)
    model_group.add_argument("--max-val-bytes", type=int)
    model_group.add_argument("--max-test-bytes", type=int)
    model_group.add_argument("--compression-sample-bytes", type=int)

    runtime_group = parser.add_argument_group("runtime overrides")
    runtime_group.add_argument("--device")
    runtime_group.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    runtime_group.add_argument("--eval-batch-size", type=int)
    runtime_group.add_argument("--output-dir")
    runtime_group.add_argument("--arithmetic-coding-mode", choices=list(DNAGPT_ARITHMETIC_CODING_MODES))
    runtime_group.add_argument("--arithmetic-merge-size", type=int)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = _resolve_config(args)
    _apply_overrides(config, args)
    validate_dnagpt_config(config)

    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))

    device = resolve_device(config.train.device)
    model, tokenizer, spec = build_dnagpt_components(config.model)
    checkpoint_path = _resolve_checkpoint_path(args, config)
    model_state, checkpoint_metadata, _ = load_dnagpt_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()
    prefix_trie = None
    if config.arithmetic.coding_mode == "base_prefix_exact_gpu_cpu":
        prefix_trie = build_dnagpt_prefix_trie(tokenizer).to(device)

    splits = load_splits(config.data, seq_length=config.model.seq_length)
    requested_splits = _normalize_splits(args.split)
    metrics = {
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_metadata.get("step"),
        "best_val_bpb": checkpoint_metadata.get("best_val_bpb"),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "resolved_config": config.to_dict(),
        "dataset": splits.summary,
        "dnagpt": {
            "variant": spec.variant,
            "kmer_size": spec.kmer_size,
            "dynamic_kmer": spec.dynamic_kmer,
            "tokenizer_vocab_size": len(tokenizer),
            "pad_id": tokenizer.pad_id,
            "unk_id": tokenizer.unk_id,
            "seq_length_tokens": config.model.seq_length,
            "max_len_tokens": spec.max_len,
            "arithmetic_coding_mode": config.arithmetic.coding_mode,
            "arithmetic_merge_size": config.arithmetic.merge_size,
        },
        "results": {},
    }

    for split_name in requested_splits:
        print(f"[compress] split={split_name} modes={','.join(args.compression_modes)}")
        metrics["results"][split_name] = _run_split(
            model=model,
            tokenizer=tokenizer,
            prefix_trie=prefix_trie,
            config=config,
            spec=spec,
            split_name=split_name,
            splits=splits,
            modes=args.compression_modes,
            device=device,
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
        export_out_dir = args.export_out_dir if args.export_out_dir is not None else args.run_dir
        _run_local_payload_export(
            run_dir=export_run_dir,
            compression_json_name=output_json.name,
            export_out_dir=export_out_dir,
            export_entity=args.export_entity,
        )


if __name__ == "__main__":
    main()
