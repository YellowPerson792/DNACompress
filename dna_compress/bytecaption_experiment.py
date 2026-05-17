from __future__ import annotations

import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .bytecaption_loader import (
    BYTECAPTION_HIDDEN_STORAGE_DTYPES,
    BYTECAPTION_LATENT_MODES,
    ByteCaptionDNACompressor,
    build_bytecaption_model,
    load_bytecaption_checkpoint,
)
from .bytecaption_tokenization import (
    BYTECAPTION_TOKENIZERS,
    ByteCaptionCacheSourceDescriptor,
    apply_bytecaption_tokenizer_to_model_config,
    build_bytecaption_tokenizer_spec,
    tokenize_bytecaption_sources_with_cache,
)
from .config import ExperimentConfig, save_experiment_config
from .data import load_splits
from .experiment import (
    autocast_context,
    build_lr_scheduler,
    init_wandb_run,
    log_wandb_metrics,
    open_training_log_file,
    resolve_device,
    save_checkpoint,
    seed_everything,
    unwrap_model,
    write_training_log_event,
)
from .nugget_data import IGNORE_INDEX, RandomNuggetWindowDataset, SequentialNuggetWindowDataset


def validate_bytecaption_config(config: ExperimentConfig) -> None:
    if config.model.implementation != "bytecaption":
        raise ValueError("ByteCaption experiment requires model.implementation='bytecaption'.")
    if config.data.nugget_tokenizer not in BYTECAPTION_TOKENIZERS:
        raise ValueError(f"data.nugget_tokenizer must be one of: {', '.join(BYTECAPTION_TOKENIZERS)}")
    if config.model.bytecaption_latent_mode not in BYTECAPTION_LATENT_MODES:
        raise ValueError(f"model.bytecaption_latent_mode must be one of: {', '.join(BYTECAPTION_LATENT_MODES)}")
    if config.model.bytecaption_hidden_storage_dtype not in BYTECAPTION_HIDDEN_STORAGE_DTYPES:
        raise ValueError(
            "model.bytecaption_hidden_storage_dtype must be one of: "
            + ", ".join(BYTECAPTION_HIDDEN_STORAGE_DTYPES)
        )
    if config.model.bytecaption_decoder_dim <= 0:
        raise ValueError("model.bytecaption_decoder_dim must be > 0.")
    if config.model.bytecaption_decoder_layers <= 0:
        raise ValueError("model.bytecaption_decoder_layers must be > 0.")
    if config.model.bytecaption_decoder_heads <= 0:
        raise ValueError("model.bytecaption_decoder_heads must be > 0.")
    if config.model.bytecaption_code_dim <= 0:
        raise ValueError("model.bytecaption_code_dim must be > 0.")
    if config.model.bytecaption_flatten_bottleneck_dim <= 0:
        raise ValueError("model.bytecaption_flatten_bottleneck_dim must be > 0.")
    if config.train.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("train.dtype must be one of: float32, float16, bfloat16")
    if config.train.init_from not in {"scratch", "pretrained", "resume"}:
        raise ValueError("train.init_from must be one of: scratch, pretrained, resume")
    if config.train.batch_size <= 0 or config.train.eval_batch_size <= 0:
        raise ValueError("train.batch_size and train.eval_batch_size must be > 0")
    if config.train.lr_scheduler not in {"none", "linear", "cosine"}:
        raise ValueError("train.lr_scheduler must be one of: none, linear, cosine")
    ratio_sum = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"data split ratios must sum to 1.0, got {ratio_sum:.6f}.")


def _storage_dtype_bytes(storage_dtype: str, runtime_dtype_name: str) -> int:
    dtype_name = runtime_dtype_name if storage_dtype == "runtime" else storage_dtype
    if dtype_name == "float32":
        return 4
    if dtype_name in {"float16", "bfloat16"}:
        return 2
    raise ValueError(f"Unsupported ByteCaption storage dtype: {dtype_name!r}")


def _latent_side_info_bits(
    *,
    model: ByteCaptionDNACompressor,
    latent_payload: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    target_bases: int,
    storage_dtype: str,
    runtime_dtype_name: str,
) -> dict[str, float]:
    dtype_bytes = _storage_dtype_bytes(storage_dtype, runtime_dtype_name)
    batch_size = int(latent_payload.shape[0])
    if model.latent_mode == "dense":
        valid = int(encoder_attention_mask.sum().item())
        hidden_bytes = valid * int(latent_payload.shape[-1]) * dtype_bytes
    elif model.latent_mode == "continuous_bottleneck":
        valid = int(encoder_attention_mask.sum().item())
        hidden_bytes = valid * int(model.code_dim) * dtype_bytes
    elif model.latent_mode == "flatten_bottleneck":
        valid = int(encoder_attention_mask.sum().item())
        hidden_bytes = batch_size * int(model.flatten_bottleneck_dim) * dtype_bytes
    else:
        raise ValueError(f"Unsupported ByteCaption latent mode: {model.latent_mode}")
    metadata_bytes = batch_size * 4
    bits = float((hidden_bytes + metadata_bytes) * 8)
    return {
        "latent_side_info_bits": bits,
        "latent_side_info_bits_per_base": bits / max(target_bases, 1),
        "bytecaption_hidden_bytes": float(hidden_bytes),
        "bytecaption_metadata_bytes": float(metadata_bytes),
        "bytecaption_valid_count": float(valid),
        "bytecaption_storage_dtype_bytes": float(dtype_bytes),
    }


def _build_source_descriptors(sources: list[bytes], splits, *, split_scope: str) -> list[ByteCaptionCacheSourceDescriptor]:
    species_summary = splits.summary["species"]
    start_key = f"{split_scope}_start"
    return [
        ByteCaptionCacheSourceDescriptor(
            species=str(item["species"]),
            source_name=str(item.get("source_name")) if item.get("source_name") is not None else None,
            payload=payload,
            clean_cache_path=str(item.get("clean_cache_path")) if item.get("clean_cache_path") is not None else None,
            source_path=str(item.get("source_path")) if item.get("source_path") is not None else None,
            split_start=int(item[start_key]) if item.get(start_key) is not None else None,
            split_length=len(payload),
        )
        for item, payload in zip(species_summary, sources)
    ]


def _target_bases(batch: dict[str, torch.Tensor]) -> int:
    valid = batch["labels"] != IGNORE_INDEX
    return int(batch["base_lengths"][valid.cpu()].sum().item())


def evaluate_bytecaption_loss(
    model: ByteCaptionDNACompressor,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
    config: ExperimentConfig,
) -> dict[str, float]:
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_bases = 0
    total_latent_bits = 0.0
    start = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device, non_blocking=True)
            valid_tokens = int((labels != IGNORE_INDEX).sum().item())
            bases = _target_bases(batch)
            with autocast_context(device, dtype_name):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_attention_mask=decoder_attention_mask,
                )
            total_nats += float(output.loss.item()) * valid_tokens
            total_tokens += valid_tokens
            total_bases += bases
            side = _latent_side_info_bits(
                model=unwrap_model(model),
                latent_payload=output.latent.payload,
                encoder_attention_mask=output.latent.attention_mask,
                target_bases=bases,
                storage_dtype=config.model.bytecaption_hidden_storage_dtype,
                runtime_dtype_name=dtype_name,
            )
            total_latent_bits += float(side["latent_side_info_bits"])

    decoder_bpb = (total_nats / max(total_bases, 1)) / math.log(2)
    latent_bpb = total_latent_bits / max(total_bases, 1)
    return {
        "loss_nats_per_token": total_nats / max(total_tokens, 1),
        "bits_per_base": decoder_bpb,
        "decoder_bits_per_base": decoder_bpb,
        "latent_side_info_bits_per_base": latent_bpb,
        "total_bits_per_base": decoder_bpb + latent_bpb,
        "tokens": float(total_tokens),
        "bases": float(total_bases),
        "elapsed_seconds": time.time() - start,
    }


def _resolve_initial_checkpoint_path(config: ExperimentConfig, mode: str, output_dir: Path) -> Path | None:
    explicit = Path(config.model.pretrained_weight_path) if config.model.pretrained_weight_path else None
    if config.train.init_from == "scratch":
        if mode == "eval":
            default = output_dir / "best.pt"
            return default if default.exists() else explicit
        return None
    if config.train.init_from == "resume":
        if explicit is not None:
            return explicit
        default = output_dir / "last.pt"
        if default.exists():
            return default
        raise FileNotFoundError("train.init_from='resume' requires a checkpoint path or output_dir/last.pt.")
    if explicit is None:
        raise FileNotFoundError("train.init_from='pretrained' requires model.pretrained_weight_path.")
    return explicit


def run_bytecaption_experiment(config: ExperimentConfig, mode: str = "all") -> dict[str, object]:
    validate_bytecaption_config(config)
    seed_everything(config.train.seed)
    device = resolve_device(config.train.device)
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_spec = build_bytecaption_tokenizer_spec(config.data, config.model)
    apply_bytecaption_tokenizer_to_model_config(config.model, tokenizer_spec)
    save_experiment_config(config, output_dir / "resolved_config.json")

    splits = load_splits(config.data)
    dataset_dir = Path(config.data.dataset_dir)
    train_sources, train_cache = tokenize_bytecaption_sources_with_cache(
        source_descriptors=_build_source_descriptors(splits.train_sources, splits, split_scope="train"),
        spec=tokenizer_spec,
        dataset_dir=dataset_dir,
        cache_enabled=config.data.clean_cache_enabled,
        cache_dir=config.data.clean_cache_dir,
        species_prefix_map=config.data.species_prefix_map,
        split_scope="train",
    )
    val_sources, val_cache = tokenize_bytecaption_sources_with_cache(
        source_descriptors=_build_source_descriptors(splits.val_sources, splits, split_scope="val"),
        spec=tokenizer_spec,
        dataset_dir=dataset_dir,
        cache_enabled=config.data.clean_cache_enabled,
        cache_dir=config.data.clean_cache_dir,
        species_prefix_map=config.data.species_prefix_map,
        split_scope="val",
    )

    train_dataset = RandomNuggetWindowDataset(
        sources=train_sources,
        seq_length=config.model.seq_length,
        samples_per_epoch=config.data.train_samples_per_epoch,
        seed=config.train.seed,
        sampling_strategy=config.data.train_sampling_strategy,
        pad_id=tokenizer_spec.pad_id,
    )
    val_dataset = SequentialNuggetWindowDataset(
        sources=val_sources,
        seq_length=config.model.seq_length,
        pad_id=tokenizer_spec.pad_id,
    )
    loader_kwargs = {
        "num_workers": config.train.num_workers,
        "pin_memory": bool(config.train.pin_memory and device.type == "cuda"),
    }
    if config.train.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.train.prefetch_factor
        loader_kwargs["persistent_workers"] = config.train.persistent_workers
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.train.eval_batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_bytecaption_model(config.model).to(device)
    checkpoint_path = _resolve_initial_checkpoint_path(config, mode, output_dir)
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader)) * max(config.train.epochs, 1))
    scheduler = build_lr_scheduler(
        optimizer,
        config.train.lr_scheduler,
        config.train.lr_warmup_steps,
        total_steps,
        config.train.lr_min_ratio,
    )
    global_step = 0
    best_val_bpb = float("inf")
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = load_bytecaption_checkpoint(model, checkpoint_path, strict=(config.train.init_from == "resume"))
        if config.train.init_from == "resume":
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            global_step = int(checkpoint.get("step", 0))
            best_val_bpb = float(checkpoint.get("best_val_bpb", best_val_bpb))

    wandb_run = init_wandb_run(config, output_dir)
    log_path, log_handle = open_training_log_file(output_dir)
    print(f"[bytecaption] train_cache={train_cache} val_cache={val_cache}")
    print(f"[bytecaption] training_log={log_path}")
    run_summary: dict[str, object] = {"output_dir": str(output_dir)}

    try:
        if mode in {"train", "all"}:
            model.train()
            for epoch in range(1, config.train.epochs + 1):
                interval_start = time.time()
                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    decoder_attention_mask = batch["decoder_attention_mask"].to(device, non_blocking=True)
                    valid_tokens = int((labels != IGNORE_INDEX).sum().item())
                    bases = _target_bases(batch)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device, config.train.dtype):
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            decoder_attention_mask=decoder_attention_mask,
                        )
                    output.loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    global_step += 1

                    if global_step % config.train.log_interval == 0:
                        elapsed = max(time.time() - interval_start, 1e-12)
                        decoder_bpb = (float(output.loss.item()) * valid_tokens / max(bases, 1)) / math.log(2)
                        side = _latent_side_info_bits(
                            model=model,
                            latent_payload=output.latent.payload,
                            encoder_attention_mask=output.latent.attention_mask,
                            target_bases=bases,
                            storage_dtype=config.model.bytecaption_hidden_storage_dtype,
                            runtime_dtype_name=config.train.dtype,
                        )
                        latent_bpb = float(side["latent_side_info_bits_per_base"])
                        event = {
                            "event": "train",
                            "epoch": epoch,
                            "step": global_step,
                            "loss_nats_per_token": float(output.loss.item()),
                            "bits_per_base": decoder_bpb,
                            "decoder_bits_per_base": decoder_bpb,
                            "latent_side_info_bits_per_base": latent_bpb,
                            "total_bits_per_base": decoder_bpb + latent_bpb,
                            "grad_norm": float(grad_norm),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "tokens_per_second": valid_tokens / elapsed,
                            "bytecaption_latent_mode": config.model.bytecaption_latent_mode,
                            "bytecaption_code_dim": config.model.bytecaption_code_dim,
                            "bytecaption_flatten_bottleneck_dim": config.model.bytecaption_flatten_bottleneck_dim,
                            **side,
                        }
                        write_training_log_event(log_handle, event)
                        log_wandb_metrics(
                            wandb_run,
                            {
                                "train/loss": float(output.loss.item()),
                                "train/bpb": decoder_bpb,
                                "train/decoder_bpb": decoder_bpb,
                                "train/latent_bpb": latent_bpb,
                                "train/total_bpb": decoder_bpb + latent_bpb,
                                "train/lr": float(optimizer.param_groups[0]["lr"]),
                                "train/grad_norm": float(grad_norm),
                            },
                            step=global_step,
                        )
                        print(
                            f"[train] epoch={epoch} step={global_step} "
                            f"loss/token={output.loss.item():.4f} bits/base={decoder_bpb:.4f} "
                            f"decoder_bpb={decoder_bpb:.4f} latent_bpb={latent_bpb:.4f} "
                            f"total_bpb={decoder_bpb + latent_bpb:.4f} "
                            f"grad_norm={float(grad_norm):.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
                        )
                        interval_start = time.time()

                    if config.train.eval_interval > 0 and global_step % config.train.eval_interval == 0:
                        metrics = evaluate_bytecaption_loss(model, val_loader, device, config.train.dtype, config)
                        write_training_log_event(log_handle, {"event": "eval", "step": global_step, **metrics})
                        log_wandb_metrics(
                            wandb_run,
                            {
                                "eval/bpb": metrics["bits_per_base"],
                                "eval/decoder_bpb": metrics["decoder_bits_per_base"],
                                "eval/latent_bpb": metrics["latent_side_info_bits_per_base"],
                                "eval/total_bpb": metrics["total_bits_per_base"],
                            },
                            step=global_step,
                        )
                        print(
                            f"[eval] step={global_step} bits/base={metrics['bits_per_base']:.4f} "
                            f"decoder_bpb={metrics['decoder_bits_per_base']:.4f} "
                            f"latent_bpb={metrics['latent_side_info_bits_per_base']:.4f} "
                            f"total_bpb={metrics['total_bits_per_base']:.4f}"
                        )
                        if metrics["bits_per_base"] < best_val_bpb:
                            best_val_bpb = float(metrics["bits_per_base"])
                            save_checkpoint(output_dir / "best.pt", model, optimizer, global_step, best_val_bpb, scheduler)
                        save_checkpoint(output_dir / "last.pt", model, optimizer, global_step, best_val_bpb, scheduler)
                        model.train()

            metrics = evaluate_bytecaption_loss(model, val_loader, device, config.train.dtype, config)
            write_training_log_event(log_handle, {"event": "eval", "step": global_step, **metrics})
            if metrics["bits_per_base"] < best_val_bpb:
                best_val_bpb = float(metrics["bits_per_base"])
                save_checkpoint(output_dir / "best.pt", model, optimizer, global_step, best_val_bpb, scheduler)
            save_checkpoint(output_dir / "last.pt", model, optimizer, global_step, best_val_bpb, scheduler)
            run_summary["best_val_bits_per_base"] = best_val_bpb

        if mode == "eval":
            metrics = evaluate_bytecaption_loss(model, val_loader, device, config.train.dtype, config)
            print(
                f"[eval] bits/base={metrics['bits_per_base']:.4f} "
                f"decoder_bpb={metrics['decoder_bits_per_base']:.4f} "
                f"latent_bpb={metrics['latent_side_info_bits_per_base']:.4f} "
                f"total_bpb={metrics['total_bits_per_base']:.4f}"
            )
            run_summary["eval"] = metrics
    finally:
        log_handle.close()
        if wandb_run is not None:
            wandb_run.finish()

    return run_summary
