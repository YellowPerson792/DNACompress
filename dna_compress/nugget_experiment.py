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
    if config.model.nugget_backbone not in NUGGET_BACKBONES:
        raise ValueError(f"model.nugget_backbone must be one of: {', '.join(NUGGET_BACKBONES)}")
    if config.data.nugget_tokenizer not in NUGGET_TOKENIZERS:
        raise ValueError(f"data.nugget_tokenizer must be one of: {', '.join(NUGGET_TOKENIZERS)}")
    if not (0.0 < config.model.nugget_ratio <= 1.0):
        raise ValueError("model.nugget_ratio must be in (0.0, 1.0].")
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


def evaluate_nugget_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
    is_distributed: bool,
) -> dict[str, float]:
    model.eval()
    total_nats = 0.0
    total_targets = 0
    total_bases = 0
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
            total_targets += int(mask.sum().item())
            total_bases += int(base_lengths[mask].sum().item())

    if is_distributed:
        reduced = torch.tensor([total_nats, float(total_targets), float(total_bases)], dtype=torch.float64, device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total_nats = float(reduced[0].item())
        total_targets = int(reduced[1].item())
        total_bases = int(reduced[2].item())

    average_nats_per_token = total_nats / max(total_targets, 1)
    average_nats_per_base = total_nats / max(total_bases, 1)
    return {
        "loss_nats_per_token": average_nats_per_token,
        "loss_nats_per_base": average_nats_per_base,
        "bits_per_token": average_nats_per_token / math.log(2),
        "bits_per_base": average_nats_per_base / math.log(2),
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
        if checkpoint_path is not None:
            if ddp.is_main_process:
                print(f"[setup] loading checkpoint {checkpoint_path}", flush=True)
            model_state, resume_metadata, raw_checkpoint = load_nugget_checkpoint(checkpoint_path, map_location="cpu")
            base_model.load_state_dict(model_state, strict=config.train.init_from == "resume")

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
        total_train_steps = config.train.epochs * len(train_loader)
        scheduler = build_lr_scheduler(
            optimizer=optimizer,
            scheduler_type=config.train.lr_scheduler,
            warmup_steps=config.train.lr_warmup_steps,
            total_steps=total_train_steps,
            min_ratio=config.train.lr_min_ratio,
        )
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
                "hidden_mode": config.model.nugget_hidden_mode,
                "hidden_storage_dtype": config.model.nugget_hidden_storage_dtype,
                "loaded_checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            },
        }
        if ddp.is_main_process and train_log_handle is not None:
            run_summary["training_log_jsonl"] = str(training_log_path)

        if mode in {"train", "all"}:
            if ddp.is_main_process:
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
                        batch_nats = float(loss.item()) * max(target_tokens, 1)
                        bits_per_base = (batch_nats / math.log(2)) / max(target_bases, 1)
                        tokens_per_second = (config.train.batch_size * config.model.seq_length * config.train.log_interval) / max(time.time() - log_started, 1e-6)
                        if ddp.is_distributed:
                            tokens_per_second *= ddp.world_size
                        grad_norm_value = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
                        event = {
                            "event": "train",
                            "step": global_step,
                            "epoch": epoch + 1,
                            "loss_nats_per_token": float(loss.item()),
                            "bits_per_base": float(bits_per_base),
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
                                "train/bpb": float(bits_per_base),
                                "train/grad_norm": grad_norm_value,
                                "train/lr": float(optimizer.param_groups[0]["lr"]),
                                "train/tokens_per_second": float(tokens_per_second),
                            },
                            step=global_step,
                        )
                        print(
                            f"[train] epoch={epoch + 1} step={global_step} "
                            f"loss/token={loss.item():.4f} bits/base={bits_per_base:.4f} "
                            f"grad_norm={grad_norm_value:.4f} "
                            f"tokens/s={tokens_per_second:.1f} lr={optimizer.param_groups[0]['lr']:.6g}",
                            flush=True,
                        )
                        log_started = time.time()

                    if global_step % config.train.eval_interval == 0:
                        if ddp.is_main_process:
                            print(f"[stage] running validation at step={global_step}...", flush=True)
                        val_metrics = evaluate_nugget_loss(model, val_loader, device, config.train.dtype, ddp.is_distributed)
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
                                        "tokens": int(val_metrics["tokens"]),
                                        "bases": int(val_metrics["bases"]),
                                    },
                                )
                            log_wandb_metrics(
                                wandb_run,
                                {"epoch": epoch + 1, "eval/loss": float(val_metrics["loss_nats_per_token"]), "eval/bpb": float(val_metrics["bits_per_base"])},
                                step=global_step,
                            )
                            _print_eval_metrics("val", val_metrics, step=global_step)
                            if val_metrics["bits_per_base"] < best_val_bpb:
                                best_val_bpb = float(val_metrics["bits_per_base"])
                                save_checkpoint(output_dir / "best.pt", unwrap_model(model), optimizer, global_step, best_val_bpb)
                        model.train()
            if best_val_bpb == float("inf"):
                if ddp.is_main_process:
                    print("[stage] running validation for checkpoint selection...", flush=True)
                val_metrics = evaluate_nugget_loss(model, val_loader, device, config.train.dtype, ddp.is_distributed)
                if ddp.is_main_process:
                    _print_eval_metrics("val", val_metrics, step=global_step)
                    best_val_bpb = float(val_metrics["bits_per_base"])
                    save_checkpoint(output_dir / "best.pt", unwrap_model(model), optimizer, global_step, best_val_bpb)
            if ddp.is_main_process:
                save_checkpoint(output_dir / "last.pt", unwrap_model(model), optimizer, global_step, best_val_bpb)
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
            val_metrics = evaluate_nugget_loss(model, val_loader, device, config.train.dtype, ddp.is_distributed)
            if ddp.is_main_process:
                _print_eval_metrics("final val", val_metrics)
                print("[stage] running final test evaluation...", flush=True)
            test_metrics = evaluate_nugget_loss(model, test_loader, device, config.train.dtype, ddp.is_distributed)
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
                        "test/loss": float(test_metrics["loss_nats_per_token"]),
                        "test/bpb": float(test_metrics["bits_per_base"]),
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
