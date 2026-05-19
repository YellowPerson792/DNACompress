from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import ExperimentConfig, save_experiment_config
from .data import load_splits
from .nugget_compression import NUGGET_ARITHMETIC_CODING_MODES, validate_nugget_hidden_policy
from .nugget_data import IGNORE_INDEX, RandomNuggetWindowDataset, SequentialNuggetWindowDataset
from .nugget_loader import (
    NUGGET_BACKBONES,
    NUGGET_LATENT_MODES,
    build_nugget_model,
    load_nugget_checkpoint,
)
from .nugget_tokenization import (
    NUGGET_TOKENIZERS,
    NuggetCacheSourceDescriptor,
    apply_nugget_tokenizer_to_model_config,
    build_nugget_tokenizer_spec,
    tokenize_nugget_sources_with_cache,
)
from .experiment import (
    autocast_context,
    build_lr_scheduler,
    cleanup_distributed,
    init_wandb_run,
    log_wandb_metrics,
    open_training_log_file,
    save_checkpoint,
    seed_everything,
    setup_distributed_context,
    unwrap_model,
    write_training_log_event,
)


def validate_nugget_config(config: ExperimentConfig) -> None:
    if config.model.implementation != "nugget":
        raise ValueError(f"Nugget experiment requires model.implementation='nugget', got {config.model.implementation!r}.")
    if not str(config.model.pretrained_weight_scope).strip():
        raise ValueError("model.pretrained_weight_scope must be a non-empty string.")
    if config.model.nugget_backbone not in NUGGET_BACKBONES:
        raise ValueError(f"model.nugget_backbone must be one of: {', '.join(NUGGET_BACKBONES)}")
    if config.data.nugget_tokenizer not in NUGGET_TOKENIZERS:
        raise ValueError(f"data.nugget_tokenizer must be one of: {', '.join(NUGGET_TOKENIZERS)}")
    if not (0.0 < config.model.nugget_ratio <= 1.0):
        raise ValueError("model.nugget_ratio must be in (0.0, 1.0].")
    if config.model.nugget_latent_mode not in NUGGET_LATENT_MODES:
        raise ValueError(f"model.nugget_latent_mode must be one of: {', '.join(NUGGET_LATENT_MODES)}")
    if config.model.nugget_code_dim <= 0:
        raise ValueError("model.nugget_code_dim must be > 0.")
    if config.model.nugget_flatten_bottleneck_dim <= 0:
        raise ValueError("model.nugget_flatten_bottleneck_dim must be > 0.")
    if config.model.nugget_scorer_layer <= 0:
        raise ValueError("model.nugget_scorer_layer must be >= 1.")
    if config.model.nugget_backbone in {"bart", "mbart"}:
        encoder_layers = config.model.nugget_bart_encoder_layers
        if encoder_layers is not None and config.model.nugget_scorer_layer > encoder_layers:
            raise ValueError("model.nugget_scorer_layer must be <= model.nugget_bart_encoder_layers.")
    if config.model.nugget_backbone == "t5" and config.model.nugget_scorer_layer > config.model.nugget_t5_num_layers:
        raise ValueError("model.nugget_scorer_layer must be <= model.nugget_t5_num_layers.")
    validate_nugget_hidden_policy(config.model.nugget_hidden_mode, config.model.nugget_hidden_storage_dtype)
    if config.train.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("train.dtype must be one of: float32, float16, bfloat16")
    if config.train.init_from not in {"scratch", "pretrained", "resume"}:
        raise ValueError("train.init_from must be one of: scratch, pretrained, resume")
    if config.train.batch_size <= 0 or config.train.eval_batch_size <= 0:
        raise ValueError("train.batch_size and train.eval_batch_size must be > 0")
    if config.train.num_workers < 0:
        raise ValueError("train.num_workers must be >= 0")
    if config.train.prefetch_factor <= 0:
        raise ValueError("train.prefetch_factor must be >= 1")
    if config.train.lr_scheduler not in {"none", "linear", "cosine"}:
        raise ValueError("train.lr_scheduler must be one of: none, linear, cosine")
    if config.data.train_sampling_strategy not in {"proportional", "uniform", "sqrt"}:
        raise ValueError("data.train_sampling_strategy must be one of: proportional, uniform, sqrt")
    ratio_sum = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"data split ratios must sum to 1.0, got {ratio_sum:.6f}.")
    if config.arithmetic.coding_mode not in NUGGET_ARITHMETIC_CODING_MODES:
        raise ValueError(
            "arithmetic.coding_mode must be one of: "
            + ", ".join(NUGGET_ARITHMETIC_CODING_MODES)
        )
    if config.arithmetic.merge_size < 1:
        raise ValueError("arithmetic.merge_size must be >= 1.")
    if config.arithmetic.coding_mode == "fixed_token_units" and config.data.nugget_tokenizer != "fixed_kmer":
        raise ValueError("fixed_token_units arithmetic requires data.nugget_tokenizer='fixed_kmer'.")
    if config.arithmetic.coding_mode == "base_prefix_exact_gpu_cpu" and config.data.nugget_tokenizer != "dnagpt_kmer":
        raise ValueError("base_prefix_exact_gpu_cpu arithmetic requires data.nugget_tokenizer='dnagpt_kmer'.")


def _species_names(splits) -> list[str]:
    return [str(item["species"]) for item in splits.summary["species"]]


def _build_nugget_cache_source_descriptors(
    sources: list[bytes],
    splits,
    *,
    split_scope: str | None = None,
) -> list[NuggetCacheSourceDescriptor]:
    species_summary = splits.summary["species"]
    if len(species_summary) != len(sources):
        raise ValueError("Nugget source descriptors do not align with loaded split sources.")
    start_key = f"{split_scope}_start" if split_scope is not None else None
    return [
        NuggetCacheSourceDescriptor(
            species=str(item["species"]),
            source_name=str(item.get("source_name")) if item.get("source_name") is not None else None,
            payload=payload,
            clean_cache_path=str(item.get("clean_cache_path")) if item.get("clean_cache_path") is not None else None,
            source_path=str(item.get("source_path")) if item.get("source_path") is not None else None,
            split_start=int(item[start_key]) if start_key is not None and item.get(start_key) is not None else None,
            split_length=len(payload),
        )
        for item, payload in zip(species_summary, sources)
    ]


def _resolve_initial_checkpoint_path(config: ExperimentConfig, mode: str, output_dir: Path) -> Path | None:
    init_from = config.train.init_from
    explicit = Path(config.model.pretrained_weight_path) if config.model.pretrained_weight_path else None
    if init_from == "scratch":
        if mode == "eval":
            default_eval_path = output_dir / "best.pt"
            if default_eval_path.exists():
                return default_eval_path
            if explicit is not None:
                return explicit
        return None
    if init_from == "resume":
        if explicit is not None:
            return explicit
        default_resume_path = output_dir / "last.pt"
        if default_resume_path.exists():
            return default_resume_path
        raise FileNotFoundError("train.init_from='resume' but no checkpoint path was provided and output_dir/last.pt does not exist.")
    if explicit is None:
        raise FileNotFoundError("train.init_from='pretrained' requires model.pretrained_weight_path.")
    return explicit


def _filter_nugget_pretrained_state(
    model_state: dict[str, torch.Tensor],
    *,
    scope: str,
    target_state_keys: set[str],
) -> dict[str, torch.Tensor]:
    scope_text = str(scope).strip()
    scope_tokens = [token.strip() for token in scope_text.split(",") if token.strip()]
    if not scope_tokens:
        raise ValueError("model.pretrained_weight_scope must not be empty.")

    # all/*: load all checkpoint tensors that also exist in the target model.
    if any(token in {"all", "*"} for token in scope_tokens):
        return {key: value for key, value in model_state.items() if key in target_state_keys}

    # auto/match: future-proof mode; loads the intersection with current model keys.
    if any(token in {"auto", "match"} for token in scope_tokens):
        return {key: value for key, value in model_state.items() if key in target_state_keys}

    selected: dict[str, torch.Tensor] = {}
    for token in scope_tokens:
        normalized = token[:-2] if token.endswith(".*") else token
        prefix = normalized if normalized.endswith(".") else f"{normalized}."
        for key, value in model_state.items():
            if key.startswith(prefix) and key in target_state_keys:
                selected[key] = value
    return selected


def _nugget_storage_dtype_bytes(hidden_storage_dtype: str, runtime_dtype_name: str) -> int:
    dtype_name = runtime_dtype_name if hidden_storage_dtype == "runtime" else hidden_storage_dtype
    if dtype_name == "float32":
        return 4
    if dtype_name in {"float16", "bfloat16"}:
        return 2
    raise ValueError(f"Unsupported Nugget hidden storage dtype for bpb estimate: {dtype_name!r}.")


def _estimate_nugget_side_info_bits(
    *,
    attention_mask: torch.Tensor,
    target_bases: int,
    nugget_ratio: float,
    hidden_dim: int,
    hidden_storage_dtype: str,
    runtime_dtype_name: str,
    requires_scores_side_info: bool,
    latent_mode: str,
    code_dim: int,
    flatten_bottleneck_dim: int,
    flatten_max_nuggets: int,
) -> dict[str, float]:
    storage_dtype_bytes = _nugget_storage_dtype_bytes(hidden_storage_dtype, runtime_dtype_name)
    valid_tokens = attention_mask.to(dtype=torch.float32).sum(dim=1)
    nugget_counts = torch.ceil(valid_tokens * float(nugget_ratio)).to(dtype=torch.long)
    valid_tokens_long = valid_tokens.to(dtype=torch.long)
    nugget_counts = torch.clamp(nugget_counts, min=1)
    nugget_counts = torch.minimum(nugget_counts, valid_tokens_long)
    nugget_count = int(nugget_counts.sum().item())
    batch_size = int(attention_mask.shape[0])
    if latent_mode == "continuous_bottleneck":
        code_bytes = nugget_count * int(code_dim) * storage_dtype_bytes
        hidden_bits = code_bytes * 8
    elif latent_mode == "flatten_bottleneck":
        code_bytes = batch_size * int(flatten_bottleneck_dim) * storage_dtype_bytes
        hidden_bits = code_bytes * 8
    else:
        code_bytes = 0
        hidden_bits = nugget_count * int(hidden_dim) * storage_dtype_bytes * 8
    score_bits = nugget_count * storage_dtype_bytes * 8 if requires_scores_side_info else 0
    metadata_bits = batch_size * 4 * 8 + score_bits
    side_info_bits = hidden_bits + metadata_bits
    return {
        "latent_side_info_bits": float(side_info_bits),
        "latent_side_info_bits_per_base": float(side_info_bits) / max(target_bases, 1),
        "nugget_hidden_bits": float(hidden_bits),
        "nugget_metadata_bits": float(metadata_bits),
        "nugget_score_bits": float(score_bits),
        "nugget_count": float(nugget_count),
        "nugget_storage_dtype_bytes": float(storage_dtype_bytes),
        "nugget_latent_mode": latent_mode,
        "nugget_code_dim": float(code_dim if latent_mode in {"continuous_bottleneck", "flatten_bottleneck"} else 0),
        "nugget_flatten_bottleneck_dim": float(flatten_bottleneck_dim if latent_mode == "flatten_bottleneck" else 0),
        "nugget_flatten_input_dim": float((flatten_max_nuggets * code_dim) if latent_mode == "flatten_bottleneck" else 0),
        "nugget_flatten_max_nuggets": float(flatten_max_nuggets if latent_mode == "flatten_bottleneck" else 0),
        "nugget_code_bytes": float(code_bytes),
    }


def evaluate_nugget_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
    is_distributed: bool,
    config: ExperimentConfig,
    hidden_dim: int,
) -> dict[str, float]:
    model.eval()
    flatten_max_nuggets = max(1, math.ceil(config.model.seq_length * config.model.nugget_ratio))
    total_nats = 0.0
    total_targets = 0
    total_bases = 0
    total_latent_side_info_bits = 0.0
    total_nugget_count = 0
    started = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device, non_blocking=True)
            base_lengths = batch["base_lengths"].to(device, non_blocking=True)
            with autocast_context(device, dtype_name):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_attention_mask=decoder_attention_mask,
                )
            losses = F.cross_entropy(
                output.logits.transpose(1, 2).float(),
                labels,
                ignore_index=IGNORE_INDEX,
                reduction="none",
            )
            mask = labels != IGNORE_INDEX
            total_nats += float(losses[mask].sum().item())
            target_tokens = int(mask.sum().item())
            target_bases = int(base_lengths[mask].sum().item())
            side_info = _estimate_nugget_side_info_bits(
                attention_mask=attention_mask,
                target_bases=target_bases,
                nugget_ratio=config.model.nugget_ratio,
                hidden_dim=hidden_dim,
                hidden_storage_dtype=config.model.nugget_hidden_storage_dtype,
                runtime_dtype_name=dtype_name,
                requires_scores_side_info=not config.model.nugget_straight_through,
                latent_mode=config.model.nugget_latent_mode,
                code_dim=config.model.nugget_code_dim,
                flatten_bottleneck_dim=config.model.nugget_flatten_bottleneck_dim,
                flatten_max_nuggets=flatten_max_nuggets,
            )
            total_targets += target_tokens
            total_bases += target_bases
            total_latent_side_info_bits += float(side_info["latent_side_info_bits"])
            total_nugget_count += int(side_info["nugget_count"])

    if is_distributed:
        reduced = torch.tensor(
            [
                total_nats,
                float(total_targets),
                float(total_bases),
                total_latent_side_info_bits,
                float(total_nugget_count),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total_nats = float(reduced[0].item())
        total_targets = int(reduced[1].item())
        total_bases = int(reduced[2].item())
        total_latent_side_info_bits = float(reduced[3].item())
        total_nugget_count = int(reduced[4].item())

    average_nats_per_token = total_nats / max(total_targets, 1)
    average_nats_per_base = total_nats / max(total_bases, 1)
    decoder_bits_per_base = average_nats_per_base / math.log(2)
    latent_side_info_bits_per_base = total_latent_side_info_bits / max(total_bases, 1)
    return {
        "loss_nats_per_token": average_nats_per_token,
        "loss_nats_per_base": average_nats_per_base,
        "bits_per_token": average_nats_per_token / math.log(2),
        "bits_per_base": decoder_bits_per_base,
        "decoder_bits_per_base": decoder_bits_per_base,
        "latent_side_info_bits": total_latent_side_info_bits,
        "latent_side_info_bits_per_base": latent_side_info_bits_per_base,
        "total_bits_per_base": decoder_bits_per_base + latent_side_info_bits_per_base,
        "nugget_count": total_nugget_count,
        "nugget_storage_dtype_bytes": _nugget_storage_dtype_bytes(config.model.nugget_hidden_storage_dtype, dtype_name),
        "nugget_latent_mode": config.model.nugget_latent_mode,
        "nugget_code_dim": config.model.nugget_code_dim if config.model.nugget_latent_mode in {"continuous_bottleneck", "flatten_bottleneck"} else 0,
        "nugget_flatten_bottleneck_dim": config.model.nugget_flatten_bottleneck_dim if config.model.nugget_latent_mode == "flatten_bottleneck" else 0,
        "nugget_flatten_input_dim": flatten_max_nuggets * config.model.nugget_code_dim if config.model.nugget_latent_mode == "flatten_bottleneck" else 0,
        "nugget_flatten_max_nuggets": flatten_max_nuggets if config.model.nugget_latent_mode == "flatten_bottleneck" else 0,
        "tokens": total_targets,
        "bases": total_bases,
        "elapsed_seconds": time.time() - started,
    }


def _print_eval_metrics(prefix: str, metrics: dict[str, float], *, step: int | None = None) -> None:
    step_fragment = f"step={step} " if step is not None else ""
    print(
        f"[eval] {prefix} {step_fragment}"
        f"loss/token={metrics['loss_nats_per_token']:.4f} "
        f"bits/base={metrics['bits_per_base']:.4f} "
        f"decoder_bpb={metrics['decoder_bits_per_base']:.4f} "
        f"latent_bpb={metrics['latent_side_info_bits_per_base']:.4f} "
        f"total_bpb={metrics['total_bits_per_base']:.4f} "
        f"bits/token={metrics['bits_per_token']:.4f} "
        f"tokens={int(metrics['tokens'])} "
        f"bases={int(metrics['bases'])} "
        f"elapsed={metrics['elapsed_seconds']:.1f}s",
        flush=True,
    )


def run_nugget_experiment(config: ExperimentConfig, mode: str = "all") -> dict[str, object]:
    validate_nugget_config(config)
    tokenizer_spec = build_nugget_tokenizer_spec(config.data, config.model)
    apply_nugget_tokenizer_to_model_config(config.model, tokenizer_spec)
    seed_everything(config.train.seed)
    ddp, device, gpu_ids = setup_distributed_context(config.train.device, config.train.gpu_ids)
    train_log_handle = None
    wandb_run = None
    try:
        output_dir = Path(config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if ddp.is_main_process:
            save_experiment_config(config, output_dir / "resolved_config.json")
            training_log_path, train_log_handle = open_training_log_file(output_dir)
            wandb_run = init_wandb_run(config, output_dir)

        if ddp.is_main_process:
            print("[setup] loading DNACorpus splits", flush=True)
        split_started = time.time()
        splits = load_splits(config.data, seq_length=config.model.seq_length)
        species_names = _species_names(splits)
        if ddp.is_main_process:
            split_entries = splits.summary["species"]
            total_train_bytes = int(sum(int(item["train_bytes"]) for item in split_entries))
            total_val_bytes = int(sum(int(item["val_bytes"]) for item in split_entries))
            total_test_bytes = int(sum(int(item["test_bytes"]) for item in split_entries))
            print(
                "[setup] DNACorpus splits loaded: "
                f"sources={len(split_entries)} "
                f"train_bytes={total_train_bytes} "
                f"val_bytes={total_val_bytes} "
                f"test_bytes={total_test_bytes} "
                f"elapsed={time.time() - split_started:.1f}s",
                flush=True,
            )
        train_descriptors = _build_nugget_cache_source_descriptors(splits.train_sources, splits, split_scope="train")
        val_descriptors = _build_nugget_cache_source_descriptors(splits.val_sources, splits, split_scope="val")
        test_descriptors = _build_nugget_cache_source_descriptors(splits.test_sources, splits, split_scope="test")
        if ddp.is_main_process:
            print(f"[setup] tokenizing {len(species_names)} source(s) with Nugget tokenizer={tokenizer_spec.name}", flush=True)
        token_cache_started = time.time()
        tokenized_train, train_token_cache_stats = tokenize_nugget_sources_with_cache(
            source_descriptors=train_descriptors,
            spec=tokenizer_spec,
            dataset_dir=Path(config.data.dataset_dir),
            cache_enabled=config.data.clean_cache_enabled,
            cache_dir=config.data.clean_cache_dir,
            species_prefix_map=config.data.species_prefix_map,
            split_scope="train",
        )
        splits.train_sources = []
        train_descriptors = []
        tokenized_val, val_token_cache_stats = tokenize_nugget_sources_with_cache(
            source_descriptors=val_descriptors,
            spec=tokenizer_spec,
            dataset_dir=Path(config.data.dataset_dir),
            cache_enabled=config.data.clean_cache_enabled,
            cache_dir=config.data.clean_cache_dir,
            species_prefix_map=config.data.species_prefix_map,
            split_scope="val",
        )
        splits.val_sources = []
        val_descriptors = []
        tokenized_test, test_token_cache_stats = tokenize_nugget_sources_with_cache(
            source_descriptors=test_descriptors,
            spec=tokenizer_spec,
            dataset_dir=Path(config.data.dataset_dir),
            cache_enabled=config.data.clean_cache_enabled,
            cache_dir=config.data.clean_cache_dir,
            species_prefix_map=config.data.species_prefix_map,
            split_scope="test",
        )
        splits.test_sources = []
        test_descriptors = []
        if ddp.is_main_process:
            total_hits = train_token_cache_stats.hits + val_token_cache_stats.hits + test_token_cache_stats.hits
            total_created = train_token_cache_stats.created + val_token_cache_stats.created + test_token_cache_stats.created
            total_rebuilt = train_token_cache_stats.rebuilt + val_token_cache_stats.rebuilt + test_token_cache_stats.rebuilt
            total_disabled = train_token_cache_stats.disabled + val_token_cache_stats.disabled + test_token_cache_stats.disabled
            total_load_seconds = train_token_cache_stats.load_seconds + val_token_cache_stats.load_seconds + test_token_cache_stats.load_seconds
            total_build_seconds = train_token_cache_stats.build_seconds + val_token_cache_stats.build_seconds + test_token_cache_stats.build_seconds
            total_write_seconds = train_token_cache_stats.write_seconds + val_token_cache_stats.write_seconds + test_token_cache_stats.write_seconds
            print(
                "[cache] nugget_tokenized "
                f"enabled={train_token_cache_stats.enabled} "
                f"dir={train_token_cache_stats.cache_dir} "
                f"hits={total_hits} "
                f"created={total_created} "
                f"rebuilt={total_rebuilt} "
                f"disabled={total_disabled} "
                f"load={total_load_seconds:.2f}s "
                f"build={total_build_seconds:.2f}s "
                f"write={total_write_seconds:.2f}s "
                f"elapsed={time.time() - token_cache_started:.1f}s",
                flush=True,
            )
        train_dataset = RandomNuggetWindowDataset(
            sources=tokenized_train,
            seq_length=config.model.seq_length,
            samples_per_epoch=config.data.train_samples_per_epoch,
            seed=config.train.seed,
            sampling_strategy=config.data.train_sampling_strategy,
            pad_id=tokenizer_spec.pad_id,
        )
        if ddp.is_main_process:
            print(
                f"[setup] training dataset ready: tokenized_sources={len(train_dataset.sources)} "
                f"samples_per_epoch={len(train_dataset)}",
                flush=True,
            )
        val_dataset = SequentialNuggetWindowDataset(
            sources=tokenized_val,
            seq_length=config.model.seq_length,
            pad_id=tokenizer_spec.pad_id,
        )
        if ddp.is_main_process:
            print(
                f"[setup] validation dataset ready: tokenized_sources={len(val_dataset.sources)} "
                f"windows={len(val_dataset)}",
                flush=True,
            )
        test_dataset = SequentialNuggetWindowDataset(
            sources=tokenized_test,
            seq_length=config.model.seq_length,
            pad_id=tokenizer_spec.pad_id,
        )
        if ddp.is_main_process:
            print(
                f"[setup] test dataset ready: tokenized_sources={len(test_dataset.sources)} "
                f"windows={len(test_dataset)}",
                flush=True,
            )

        dataloader_pin_memory = bool(config.train.pin_memory and device.type == "cuda")
        dataloader_persistent_workers = bool(config.train.persistent_workers and config.train.num_workers > 0)
        common_kwargs: dict[str, object] = {
            "num_workers": config.train.num_workers,
            "pin_memory": dataloader_pin_memory,
            "persistent_workers": dataloader_persistent_workers,
        }
        if config.train.num_workers > 0:
            common_kwargs["prefetch_factor"] = config.train.prefetch_factor
        train_sampler = DistributedSampler(train_dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=True) if ddp.is_distributed else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False) if ddp.is_distributed else None
        test_sampler = DistributedSampler(test_dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False) if ddp.is_distributed else None
        train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=train_sampler is None, sampler=train_sampler, **common_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=config.train.eval_batch_size, shuffle=False, sampler=val_sampler, **common_kwargs)
        test_loader = DataLoader(test_dataset, batch_size=config.train.eval_batch_size, shuffle=False, sampler=test_sampler, **common_kwargs)

        if ddp.is_main_process:
            print(
                f"[setup] building Nugget model backbone={config.model.nugget_backbone} "
                f"vocab_size={tokenizer_spec.vocab_size}",
                flush=True,
            )
        base_model, backbone_spec = build_nugget_model(config.model, tokenizer_spec)
        parameter_count = int(sum(parameter.numel() for parameter in base_model.parameters()))
        if ddp.is_main_process:
            print(f"[setup] built Nugget model parameters={parameter_count:,}", flush=True)
        checkpoint_path = _resolve_initial_checkpoint_path(config, mode, output_dir)
        resume_metadata: dict[str, object] = {}
        raw_checkpoint: dict[str, object] | None = None
        optimizer_state_restored = False
        scheduler_state_restored = False
        if checkpoint_path is not None:
            if ddp.is_main_process:
                print(f"[setup] loading checkpoint {checkpoint_path}", flush=True)
            model_state, resume_metadata, raw_checkpoint = load_nugget_checkpoint(checkpoint_path, map_location="cpu")
            if config.train.init_from == "pretrained":
                model_state = _filter_nugget_pretrained_state(
                    model_state,
                    scope=config.model.pretrained_weight_scope,
                    target_state_keys=set(base_model.state_dict().keys()),
                )
            load_result = base_model.load_state_dict(model_state, strict=config.train.init_from == "resume")
            if ddp.is_main_process and config.train.init_from == "pretrained":
                print(
                    "[setup] pretrained weights loaded: "
                    f"scope={config.model.pretrained_weight_scope} "
                    f"tensors={len(model_state)} "
                    f"missing={len(load_result.missing_keys)} "
                    f"unexpected={len(load_result.unexpected_keys)}",
                    flush=True,
                )

        if ddp.is_main_process:
            print(f"[setup] moving model to device={device}", flush=True)
        base_model = base_model.to(device)
        model: torch.nn.Module = base_model
        if ddp.is_distributed:
            model = DistributedDataParallel(base_model, device_ids=[device.index], output_device=device.index)
        optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
        if config.train.init_from == "resume" and isinstance(raw_checkpoint, dict):
            optimizer_state = raw_checkpoint.get("optimizer_state")
            if isinstance(optimizer_state, dict):
                optimizer.load_state_dict(optimizer_state)
                optimizer_state_restored = True
        total_train_steps = config.train.epochs * len(train_loader)
        scheduler = build_lr_scheduler(
            optimizer=optimizer,
            scheduler_type=config.train.lr_scheduler,
            warmup_steps=config.train.lr_warmup_steps,
            total_steps=total_train_steps,
            min_ratio=config.train.lr_min_ratio,
        )
        if config.train.init_from == "resume" and isinstance(raw_checkpoint, dict):
            scheduler_state = raw_checkpoint.get("scheduler_state")
            if scheduler is not None and isinstance(scheduler_state, dict):
                scheduler.load_state_dict(scheduler_state)
                scheduler_state_restored = True
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and config.train.dtype == "float16")
        best_val_bpb = float(resume_metadata.get("best_val_bpb", float("inf"))) if config.train.init_from == "resume" else float("inf")
        global_step = int(resume_metadata.get("step", 0)) if config.train.init_from == "resume" else 0

        run_summary: dict[str, object] = {
            "device": str(device),
            "gpu_ids": gpu_ids,
            "distributed_data_parallel": ddp.is_distributed,
            "world_size": ddp.world_size,
            "model_parameters": parameter_count,
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
                "value_ffn": config.model.nugget_value_ffn,
                "value_ffn_layer_norm": config.model.nugget_value_ffn_layer_norm,
                "hidden_mode": config.model.nugget_hidden_mode,
                "hidden_storage_dtype": config.model.nugget_hidden_storage_dtype,
                "latent_mode": config.model.nugget_latent_mode,
                "bottleneck_layer_norm": config.model.nugget_bottleneck_layer_norm,
                "flatten_bottleneck_dim": config.model.nugget_flatten_bottleneck_dim,
                "flatten_input_dim": max(1, math.ceil(config.model.seq_length * config.model.nugget_ratio)) * config.model.nugget_code_dim,
                "flatten_max_nuggets": max(1, math.ceil(config.model.seq_length * config.model.nugget_ratio)),
                "code_dim": config.model.nugget_code_dim,
                "loaded_checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "pretrained_weight_scope": config.model.pretrained_weight_scope,
                "optimizer_state_restored": optimizer_state_restored,
                "scheduler_state_restored": scheduler_state_restored,
            },
        }
        if ddp.is_main_process and train_log_handle is not None:
            run_summary["training_log_jsonl"] = str(training_log_path)
        if ddp.is_main_process and checkpoint_path is not None:
            print(
                f"[startup] loaded checkpoint={checkpoint_path} init_from={config.train.init_from} "
                f"optimizer_restored={optimizer_state_restored} "
                f"scheduler_restored={scheduler_state_restored}",
                flush=True,
            )

        if mode in {"train", "all"}:
            if ddp.is_main_process:
                print("[setup] model architecture:", flush=True)
                print(base_model, flush=True)
                print(f"[setup] starting training epochs={config.train.epochs} steps_per_epoch={len(train_loader)}", flush=True)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            log_started = time.time()
            for epoch in range(config.train.epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                for batch in train_loader:
                    global_step += 1
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    decoder_attention_mask = batch["decoder_attention_mask"].to(device, non_blocking=True)
                    base_lengths = batch["base_lengths"].to(device, non_blocking=True)
                    with autocast_context(device, config.train.dtype):
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            decoder_attention_mask=decoder_attention_mask,
                        )
                        loss = output.loss
                        decoder_loss = output.decoder_loss
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    if global_step % config.train.log_interval == 0 and ddp.is_main_process:
                        valid_mask = labels != IGNORE_INDEX
                        target_tokens = int(valid_mask.sum().item())
                        target_bases = int(base_lengths[valid_mask].sum().item())
                        decoder_loss_value = float(decoder_loss.detach().float().item())
                        batch_nats = decoder_loss_value * max(target_tokens, 1)
                        bits_per_base = (batch_nats / math.log(2)) / max(target_bases, 1)
                        side_info = _estimate_nugget_side_info_bits(
                            attention_mask=attention_mask,
                            target_bases=target_bases,
                            nugget_ratio=config.model.nugget_ratio,
                            hidden_dim=backbone_spec.d_model,
                            hidden_storage_dtype=config.model.nugget_hidden_storage_dtype,
                            runtime_dtype_name=config.train.dtype,
                            requires_scores_side_info=not config.model.nugget_straight_through,
                            latent_mode=config.model.nugget_latent_mode,
                            code_dim=config.model.nugget_code_dim,
                            flatten_bottleneck_dim=config.model.nugget_flatten_bottleneck_dim,
                            flatten_max_nuggets=max(1, math.ceil(config.model.seq_length * config.model.nugget_ratio)),
                        )
                        latent_side_info_bits_per_base = float(side_info["latent_side_info_bits_per_base"])
                        total_bits_per_base = bits_per_base + latent_side_info_bits_per_base
                        tokens_per_second = (config.train.batch_size * config.model.seq_length * config.train.log_interval) / max(time.time() - log_started, 1e-6)
                        if ddp.is_distributed:
                            tokens_per_second *= ddp.world_size
                        grad_norm_value = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
                        event = {
                            "event": "train",
                            "step": global_step,
                            "epoch": epoch + 1,
                            "loss_nats_per_token": float(loss.item()),
                            "decoder_loss_nats_per_token": decoder_loss_value,
                            "bits_per_base": float(bits_per_base),
                            "decoder_bits_per_base": float(bits_per_base),
                            "latent_side_info_bits_per_base": latent_side_info_bits_per_base,
                            "total_bits_per_base": float(total_bits_per_base),
                            "latent_side_info_bits": float(side_info["latent_side_info_bits"]),
                            "nugget_count": int(side_info["nugget_count"]),
                            "nugget_storage_dtype_bytes": int(side_info["nugget_storage_dtype_bytes"]),
                            "nugget_latent_mode": config.model.nugget_latent_mode,
                            "nugget_code_dim": int(side_info["nugget_code_dim"]),
                            "nugget_flatten_bottleneck_dim": int(side_info["nugget_flatten_bottleneck_dim"]),
                            "nugget_flatten_input_dim": int(side_info["nugget_flatten_input_dim"]),
                            "nugget_flatten_max_nuggets": int(side_info["nugget_flatten_max_nuggets"]),
                            "nugget_code_bytes": int(side_info["nugget_code_bytes"]),
                            "grad_norm": grad_norm_value,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "tokens_per_second": float(tokens_per_second),
                            "tokens": target_tokens,
                            "bases": target_bases,
                        }
                        if train_log_handle is not None:
                            write_training_log_event(train_log_handle, event)
                        log_wandb_metrics(
                            wandb_run,
                            {
                                "epoch": epoch + 1,
                                "train/loss": float(loss.item()),
                                "train/decoder_loss": decoder_loss_value,
                                "train/bpb": float(bits_per_base),
                                "train/decoder_bpb": float(bits_per_base),
                                "train/latent_bpb": latent_side_info_bits_per_base,
                                "train/total_bpb": float(total_bits_per_base),
                                "train/nugget_flatten_bottleneck_dim": float(side_info["nugget_flatten_bottleneck_dim"]),
                                "train/nugget_flatten_input_dim": float(side_info["nugget_flatten_input_dim"]),
                                "train/nugget_flatten_max_nuggets": float(side_info["nugget_flatten_max_nuggets"]),
                                "train/grad_norm": grad_norm_value,
                                "train/lr": float(optimizer.param_groups[0]["lr"]),
                                "train/tokens_per_second": float(tokens_per_second),
                            },
                            step=global_step,
                        )
                        print(
                            f"[train] epoch={epoch + 1} step={global_step} "
                            f"loss/token={loss.item():.4f} decoder_loss={decoder_loss_value:.4f} "
                            f"bits/base={bits_per_base:.4f} "
                            f"decoder_bpb={bits_per_base:.4f} "
                            f"latent_bpb={latent_side_info_bits_per_base:.4f} "
                            f"total_bpb={total_bits_per_base:.4f} "
                            f"grad_norm={grad_norm_value:.4f} "
                            f"tokens/s={tokens_per_second:.1f} lr={optimizer.param_groups[0]['lr']:.6g}",
                            flush=True,
                        )
                        log_started = time.time()

                    if global_step % config.train.eval_interval == 0:
                        if ddp.is_main_process:
                            print(f"[stage] running validation at step={global_step}...", flush=True)
                        val_metrics = evaluate_nugget_loss(
                            model,
                            val_loader,
                            device,
                            config.train.dtype,
                            ddp.is_distributed,
                            config,
                            backbone_spec.d_model,
                        )
                        if ddp.is_main_process:
                            if train_log_handle is not None:
                                write_training_log_event(
                                    train_log_handle,
                                    {
                                        "event": "eval",
                                        "split": "val",
                                        "step": global_step,
                                        "epoch": epoch + 1,
                                        "loss_nats_per_token": float(val_metrics["loss_nats_per_token"]),
                                        "bits_per_base": float(val_metrics["bits_per_base"]),
                                        "decoder_bits_per_base": float(val_metrics["decoder_bits_per_base"]),
                                        "latent_side_info_bits_per_base": float(val_metrics["latent_side_info_bits_per_base"]),
                                        "total_bits_per_base": float(val_metrics["total_bits_per_base"]),
                                        "latent_side_info_bits": float(val_metrics["latent_side_info_bits"]),
                                        "nugget_count": int(val_metrics["nugget_count"]),
                                        "nugget_storage_dtype_bytes": int(val_metrics["nugget_storage_dtype_bytes"]),
                                        "nugget_latent_mode": val_metrics["nugget_latent_mode"],
                                        "nugget_code_dim": int(val_metrics["nugget_code_dim"]),
                                        "nugget_flatten_bottleneck_dim": int(val_metrics["nugget_flatten_bottleneck_dim"]),
                                        "nugget_flatten_input_dim": int(val_metrics["nugget_flatten_input_dim"]),
                                        "nugget_flatten_max_nuggets": int(val_metrics["nugget_flatten_max_nuggets"]),
                                        "tokens": int(val_metrics["tokens"]),
                                        "bases": int(val_metrics["bases"]),
                                    },
                                )
                            log_wandb_metrics(
                                wandb_run,
                                {
                                    "epoch": epoch + 1,
                                    "eval/loss": float(val_metrics["loss_nats_per_token"]),
                                    "eval/bpb": float(val_metrics["bits_per_base"]),
                                    "eval/decoder_bpb": float(val_metrics["decoder_bits_per_base"]),
                                    "eval/latent_bpb": float(val_metrics["latent_side_info_bits_per_base"]),
                                    "eval/total_bpb": float(val_metrics["total_bits_per_base"]),
                                    "eval/nugget_flatten_bottleneck_dim": float(val_metrics["nugget_flatten_bottleneck_dim"]),
                                    "eval/nugget_flatten_input_dim": float(val_metrics["nugget_flatten_input_dim"]),
                                    "eval/nugget_flatten_max_nuggets": float(val_metrics["nugget_flatten_max_nuggets"]),
                                },
                                step=global_step,
                            )
                            _print_eval_metrics("val", val_metrics, step=global_step)
                            if val_metrics["bits_per_base"] < best_val_bpb:
                                best_val_bpb = float(val_metrics["bits_per_base"])
                                save_checkpoint(
                                    output_dir / "best.pt",
                                    unwrap_model(model),
                                    optimizer,
                                    global_step,
                                    best_val_bpb,
                                    scheduler,
                                )
                        model.train()
            if best_val_bpb == float("inf"):
                if ddp.is_main_process:
                    print("[stage] running validation for checkpoint selection...", flush=True)
                val_metrics = evaluate_nugget_loss(
                    model,
                    val_loader,
                    device,
                    config.train.dtype,
                    ddp.is_distributed,
                    config,
                    backbone_spec.d_model,
                )
                if ddp.is_main_process:
                    _print_eval_metrics("val", val_metrics, step=global_step)
                    best_val_bpb = float(val_metrics["bits_per_base"])
                    save_checkpoint(
                        output_dir / "best.pt",
                        unwrap_model(model),
                        optimizer,
                        global_step,
                        best_val_bpb,
                        scheduler,
                    )
            if ddp.is_main_process:
                save_checkpoint(
                    output_dir / "last.pt",
                    unwrap_model(model),
                    optimizer,
                    global_step,
                    best_val_bpb,
                    scheduler,
                )
                run_summary["best_val_bits_per_base"] = best_val_bpb

        if ddp.is_distributed:
            dist.barrier()
        best_checkpoint_path = output_dir / "best.pt"
        if mode in {"train", "all"} and best_checkpoint_path.exists():
            model_state, _, _ = load_nugget_checkpoint(best_checkpoint_path, map_location=device)
            unwrap_model(model).load_state_dict(model_state)

        if mode in {"eval", "all"}:
            if ddp.is_main_process:
                print("[stage] running final validation...", flush=True)
            val_metrics = evaluate_nugget_loss(
                model,
                val_loader,
                device,
                config.train.dtype,
                ddp.is_distributed,
                config,
                backbone_spec.d_model,
            )
            if ddp.is_main_process:
                _print_eval_metrics("final val", val_metrics)
                print("[stage] running final test evaluation...", flush=True)
            test_metrics = evaluate_nugget_loss(
                model,
                test_loader,
                device,
                config.train.dtype,
                ddp.is_distributed,
                config,
                backbone_spec.d_model,
            )
            if ddp.is_main_process:
                _print_eval_metrics("final test", test_metrics)
                run_summary["validation"] = val_metrics
                run_summary["test"] = test_metrics
                if train_log_handle is not None:
                    write_training_log_event(train_log_handle, {"event": "eval", "split": "val", "step": global_step, "is_final": True, **val_metrics})
                    write_training_log_event(train_log_handle, {"event": "eval", "split": "test", "step": global_step, "is_final": True, **test_metrics})
                log_wandb_metrics(
                    wandb_run,
                    {
                        "eval/final_loss": float(val_metrics["loss_nats_per_token"]),
                        "eval/final_bpb": float(val_metrics["bits_per_base"]),
                        "eval/final_decoder_bpb": float(val_metrics["decoder_bits_per_base"]),
                        "eval/final_latent_bpb": float(val_metrics["latent_side_info_bits_per_base"]),
                        "eval/final_total_bpb": float(val_metrics["total_bits_per_base"]),
                        "test/loss": float(test_metrics["loss_nats_per_token"]),
                        "test/bpb": float(test_metrics["bits_per_base"]),
                        "test/decoder_bpb": float(test_metrics["decoder_bits_per_base"]),
                        "test/latent_bpb": float(test_metrics["latent_side_info_bits_per_base"]),
                        "test/total_bpb": float(test_metrics["total_bits_per_base"]),
                    },
                    step=global_step,
                )

        if ddp.is_main_process:
            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            if wandb_run is not None:
                wandb_run.summary["output_dir"] = str(output_dir)
                wandb_run.summary["model_parameters"] = run_summary["model_parameters"]
            return run_summary
        return {}
    finally:
        if train_log_handle is not None:
            train_log_handle.close()
        if wandb_run is not None:
            wandb_run.finish()
        if ddp.is_distributed:
            cleanup_distributed()
