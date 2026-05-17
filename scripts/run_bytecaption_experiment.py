from __future__ import annotations

"""Train or evaluate ByteCaption/PureT DNA compression models.

Examples:

  # Train/evaluate a ByteCaption encoder-decoder with dense ByteFormer latents.
  python scripts/run_bytecaption_experiment.py \
    --config configs/dna_bytecaption_quick.json \
    --mode all \
    --implementation bytecaption \
    --bytecaption-tokenizer fixed_kmer \
    --token-merge-size 3 \
    --token-merge-alphabet ACGTN \
    --bytecaption-latent-mode flatten_bottleneck \
    --bytecaption-flatten-bottleneck-dim 512 \
    --bytecaption-code-dim 64 \
    --bytecaption-decoder-dim 512 \
    --bytecaption-decoder-layers 3 \
    --bytecaption-decoder-heads 8 \
    --bytecaption-decoder-dropout 0 \
    --bytecaption-decoder-ff-dropout 0 \
    --bytecaption-hidden-storage-dtype runtime \
    --seq-length 1024 \
    --dataset-dir datasets/DNACorpus \
    --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
    --train-ratio 0.6 \
    --val-ratio 0.2 \
    --test-ratio 0.2 \
    --train-samples-per-epoch 600000 \
    --compression-sample-bytes 100000 \
    --arithmetic-coding-mode model_symbol \
    --arithmetic-merge-size 1 \
    --device cuda \
    --dtype bfloat16 \
    --epochs 1 \
    --batch-size 32 \
    --eval-batch-size 32 \
    --learning-rate 2e-4 \
    --weight-decay 0 \
    --lr-scheduler cosine \
    --lr-warmup-steps 500 \
    --lr-min-ratio 0.1 \
    --grad-clip-norm 1.0 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --persistent-workers \
    --pin-memory \
    --log-interval 25 \
    --eval-interval 2000 \
    --print-config \
    --wandb-project dna-compress \
    --wandb-name dna_bytecaption_flatten_d64_512

  # Train/evaluate a smaller flatten-bottleneck run.
  python scripts/run_bytecaption_experiment.py \
    --config configs/dna_bytecaption_quick.json \
    --mode all \
    --bytecaption-tokenizer fixed_kmer \
    --token-merge-size 6 \
    --token-merge-alphabet ACGTN \
    --bytecaption-latent-mode flatten_bottleneck \
    --bytecaption-bottleneck-layer-norm \
    --bytecaption-code-dim 64 \
    --bytecaption-flatten-bottleneck-dim 512 \
    --bytecaption-hidden-storage-dtype float16 \
    --seq-length 512 \
    --dataset-dir datasets/DNACorpus \
    --species HoSa \
    --train-ratio 0.9 \
    --val-ratio 0.05 \
    --test-ratio 0.05 \
    --train-samples-per-epoch 4096 \
    --compression-sample-bytes 16384 \
    --arithmetic-coding-mode fixed_token_units \
    --arithmetic-merge-size 1 \
    --device cuda \
    --dtype float16 \
    --epochs 1 \
    --batch-size 8 \
    --eval-batch-size 8 \
    --learning-rate 5e-5 \
    --num-workers 2 \
    --prefetch-factor 2 \
    --persistent-workers \
    --pin-memory \
    --output-dir outputs/dna_bytecaption_flatten_quick \
    --wandb-name dna_bytecaption_flatten_quick

  # Evaluate an existing checkpoint only.
  python scripts/run_bytecaption_experiment.py \
    --config outputs/dna_bytecaption_dense_d64/resolved_config.json \
    --mode eval \
    --init-from pretrained \
    --pretrained-weight-path outputs/dna_bytecaption_dense_d64/best.pt \
    --pretrained-weight-scope all \
    --device cuda \
    --dtype bfloat16 \
    --eval-batch-size 32 \
    --output-dir outputs/dna_bytecaption_dense_d64_eval \
    --print-config
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

from dna_compress.bytecaption_experiment import run_bytecaption_experiment, validate_bytecaption_config
from dna_compress.bytecaption_loader import BYTECAPTION_HIDDEN_STORAGE_DTYPES, BYTECAPTION_LATENT_MODES
from dna_compress.bytecaption_tokenization import BYTECAPTION_TOKENIZERS, apply_bytecaption_tokenizer_to_model_config, build_bytecaption_tokenizer_spec
from dna_compress.config import load_experiment_config


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


def _apply_overrides(config: Any, args: argparse.Namespace) -> None:
    if args.species is not None:
        config.data.species = args.species
    _apply_if_not_none(config, "model.implementation", args.implementation)
    _apply_if_not_none(config, "model.pretrained_weight_path", args.pretrained_weight_path)
    _apply_if_not_none(config, "model.pretrained_weight_scope", args.pretrained_weight_scope)
    _apply_if_not_none(config, "model.seq_length", args.seq_length)
    _apply_if_not_none(config, "model.bytecaption_decoder_dim", args.bytecaption_decoder_dim)
    _apply_if_not_none(config, "model.bytecaption_decoder_layers", args.bytecaption_decoder_layers)
    _apply_if_not_none(config, "model.bytecaption_decoder_heads", args.bytecaption_decoder_heads)
    _apply_if_not_none(config, "model.bytecaption_decoder_dropout", args.bytecaption_decoder_dropout)
    _apply_if_not_none(config, "model.bytecaption_decoder_ff_dropout", args.bytecaption_decoder_ff_dropout)
    _apply_if_not_none(config, "model.bytecaption_byteformer_config_path", args.bytecaption_byteformer_config_path)
    _apply_if_not_none(config, "model.bytecaption_byteformer_weight_path", args.bytecaption_byteformer_weight_path)
    _apply_if_not_none(config, "model.bytecaption_latent_mode", args.bytecaption_latent_mode)
    _apply_if_not_none(config, "model.bytecaption_code_dim", args.bytecaption_code_dim)
    _apply_if_not_none(config, "model.bytecaption_flatten_bottleneck_dim", args.bytecaption_flatten_bottleneck_dim)
    _apply_if_not_none(config, "model.bytecaption_bottleneck_layer_norm", args.bytecaption_bottleneck_layer_norm)
    _apply_if_not_none(config, "model.bytecaption_hidden_storage_dtype", args.bytecaption_hidden_storage_dtype)
    _apply_if_not_none(config, "data.dataset_dir", args.dataset_dir)
    _apply_if_not_none(config, "data.nugget_tokenizer", args.bytecaption_tokenizer)
    _apply_if_not_none(config, "data.sequence_source_mode", args.sequence_source_mode)
    _apply_if_not_none(config, "data.multi_sequence_mode", args.multi_sequence_mode)
    _apply_if_not_none(config, "data.train_ratio", args.train_ratio)
    _apply_if_not_none(config, "data.val_ratio", args.val_ratio)
    _apply_if_not_none(config, "data.test_ratio", args.test_ratio)
    _apply_if_not_none(config, "data.train_samples_per_epoch", args.train_samples_per_epoch)
    _apply_if_not_none(config, "data.token_merge_size", args.token_merge_size)
    _apply_if_not_none(config, "data.token_merge_alphabet", args.token_merge_alphabet)
    _apply_if_not_none(config, "data.compression_sample_bytes", args.compression_sample_bytes)
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
    _apply_if_not_none(config, "output.output_dir", args.output_dir)
    _apply_if_not_none(config, "output.wandb_project", args.wandb_project)
    _apply_if_not_none(config, "output.wandb_name", args.wandb_name)
    _apply_if_not_none(config, "arithmetic.coding_mode", args.arithmetic_coding_mode)
    _apply_if_not_none(config, "arithmetic.merge_size", args.arithmetic_merge_size)
    for item in args.override:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected section.key=value")
        key, raw_value = item.split("=", 1)
        _set_nested_attr(config, key.strip(), _parse_scalar(raw_value.strip()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dna_bytecaption_quick.json")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--implementation", default=None)
    parser.add_argument("--pretrained-weight-path", default=None)
    parser.add_argument("--pretrained-weight-scope", default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--bytecaption-decoder-dim", type=int, default=None)
    parser.add_argument("--bytecaption-decoder-layers", type=int, default=None)
    parser.add_argument("--bytecaption-decoder-heads", type=int, default=None)
    parser.add_argument("--bytecaption-decoder-dropout", type=float, default=None)
    parser.add_argument("--bytecaption-decoder-ff-dropout", type=float, default=None)
    parser.add_argument("--bytecaption-byteformer-config-path", default=None)
    parser.add_argument("--bytecaption-byteformer-weight-path", default=None)
    parser.add_argument("--bytecaption-latent-mode", choices=BYTECAPTION_LATENT_MODES, default=None)
    parser.add_argument("--bytecaption-code-dim", type=int, default=None)
    parser.add_argument("--bytecaption-flatten-bottleneck-dim", type=int, default=None)
    parser.add_argument("--bytecaption-bottleneck-layer-norm", action="store_true", default=None)
    parser.add_argument("--bytecaption-hidden-storage-dtype", choices=BYTECAPTION_HIDDEN_STORAGE_DTYPES, default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--species", nargs="+", default=None)
    parser.add_argument("--bytecaption-tokenizer", choices=BYTECAPTION_TOKENIZERS, default=None)
    parser.add_argument("--sequence-source-mode", choices=["auto", "single", "multi"], default=None)
    parser.add_argument("--multi-sequence-mode", choices=["separate", "concat"], default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--train-samples-per-epoch", type=int, default=None)
    parser.add_argument("--token-merge-size", type=int, default=None)
    parser.add_argument("--token-merge-alphabet", default=None)
    parser.add_argument("--compression-sample-bytes", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--init-from", choices=["scratch", "pretrained", "resume"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr-scheduler", choices=["none", "linear", "cosine"], default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument("--lr-min-ratio", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", action="store_true", default=None)
    parser.add_argument("--pin-memory", action="store_true", default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--arithmetic-coding-mode", choices=["model_symbol", "fixed_token_units"], default=None)
    parser.add_argument("--arithmetic-merge-size", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    _apply_overrides(config, args)
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output.output_dir = str(Path(config.output.output_dir).with_name(f"{Path(config.output.output_dir).name}_{stamp}"))
    spec = build_bytecaption_tokenizer_spec(config.data, config.model)
    apply_bytecaption_tokenizer_to_model_config(config.model, spec)
    validate_bytecaption_config(config)
    if args.print_config or args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    run_bytecaption_experiment(config, mode=args.mode)


if __name__ == "__main__":
    main()
