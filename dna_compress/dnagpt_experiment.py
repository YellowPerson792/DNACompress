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
from .dnagpt_data import IGNORE_INDEX, RandomDNAGPTWindowDataset, SequentialDNAGPTWindowDataset
from .dnagpt_loader import (
    build_dnagpt_components,
    default_pretrained_weight_path,
    get_variant_spec,
    load_dnagpt_checkpoint,
)
from .dnagpt_tokenization import tokenize_dna_sources
from .experiment import (
    autocast_context,
    build_lr_scheduler,
    cleanup_distributed,
    save_checkpoint,
    seed_everything,
    setup_distributed_context,
    unwrap_model,
)


def validate_dnagpt_config(config: ExperimentConfig) -> None:
    if config.model.implementation != "dnagpt":
        raise ValueError(
            f"DNAGPT experiment requires model.implementation='dnagpt', got '{config.model.implementation}'."
        )

    spec = get_variant_spec(config.model.variant)
    if config.model.seq_length <= 0:
        raise ValueError("model.seq_length must be > 0 for DNAGPT.")
    if config.model.seq_length > spec.max_len:
        raise ValueError(
            f"model.seq_length ({config.model.seq_length}) exceeds {config.model.variant} max_len ({spec.max_len})."
        )

    ratio_sum = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"data split ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"(train={config.data.train_ratio}, val={config.data.val_ratio}, test={config.data.test_ratio})."
        )

    if config.data.token_merge_size != 1:
        raise ValueError("DNAGPT does not use data.token_merge_size. Keep it set to 1.")

    if config.train.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("train.dtype must be one of: float32, float16, bfloat16")
    if config.train.batch_size <= 0 or config.train.eval_batch_size <= 0:
        raise ValueError("train.batch_size and train.eval_batch_size must be > 0")
    if config.train.init_from not in {"scratch", "pretrained", "resume"}:
        raise ValueError("train.init_from must be one of: scratch, pretrained, resume")
    if config.data.train_sampling_strategy not in {"proportional", "uniform", "sqrt"}:
        raise ValueError("data.train_sampling_strategy must be one of: proportional, uniform, sqrt")
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


def _species_names(splits) -> list[str]:
    return [str(item["species"]) for item in splits.summary["species"]]


def _build_tokenized_split_sources(config: ExperimentConfig, splits, tokenizer, spec):
    species_names = _species_names(splits)
    common_kwargs = dict(
        species_names=species_names,
        tokenizer=tokenizer,
        kmer_size=spec.kmer_size,
        species_prefix_map=config.data.species_prefix_map,
    )
    return {
        "train": tokenize_dna_sources(sources=splits.train_sources, **common_kwargs),
        "val": tokenize_dna_sources(sources=splits.val_sources, **common_kwargs),
        "test": tokenize_dna_sources(sources=splits.test_sources, **common_kwargs),
    }


def _dataset_token_summary(tokenized_sources) -> dict[str, object]:
    per_species: list[dict[str, object]] = []
    total_tokens = 0
    total_bases = 0
    for source in tokenized_sources:
        token_count = len(source.dna_token_ids)
        total_tokens += token_count
        total_bases += source.total_bases
        per_species.append(
            {
                "species": source.species,
                "prefix_token": source.prefix_token,
                "prefix_token_count": len(source.prefix_ids),
                "dna_token_count": token_count,
                "total_bases": source.total_bases,
            }
        )
    return {
        "total_dna_token_count": total_tokens,
        "total_bases": total_bases,
        "per_species": per_species,
    }


def evaluate_dnagpt_loss(
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
            labels = batch["labels"].to(device, non_blocking=True)
            base_lengths = batch["base_lengths"].to(device, non_blocking=True)
            with autocast_context(device, dtype_name):
                logits = model(input_ids)
            losses = F.cross_entropy(
                logits.transpose(1, 2).float(),
                labels,
                ignore_index=IGNORE_INDEX,
                reduction="none",
            )
            mask = labels != IGNORE_INDEX
            total_nats += float(losses[mask].sum().item())
            total_targets += int(mask.sum().item())
            total_bases += int(base_lengths[mask].sum().item())

    if is_distributed:
        reduced = torch.tensor(
            [total_nats, float(total_targets), float(total_bases)],
            dtype=torch.float64,
            device=device,
        )
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
        raise FileNotFoundError(
            "train.init_from='resume' but no checkpoint path was provided and output_dir/last.pt does not exist."
        )

    if explicit is not None:
        return explicit
    return default_pretrained_weight_path(config.model.variant)


def run_dnagpt_experiment(config: ExperimentConfig, mode: str = "all") -> dict[str, object]:
    validate_dnagpt_config(config)
    seed_everything(config.train.seed)
    ddp, device, gpu_ids = setup_distributed_context(config.train.device, config.train.gpu_ids)
    try:
        output_dir = Path(config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if ddp.is_main_process:
            save_experiment_config(config, output_dir / "resolved_config.json")

        base_model, tokenizer, spec = build_dnagpt_components(config.model)
        checkpoint_path = _resolve_initial_checkpoint_path(config, mode, output_dir)
        resume_metadata: dict[str, object] = {}
        raw_checkpoint: dict[str, object] | None = None
        if checkpoint_path is not None:
            model_state, resume_metadata, raw_checkpoint = load_dnagpt_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )
            base_model.load_state_dict(model_state, strict=False)

        base_model = base_model.to(device)
        model: torch.nn.Module = base_model
        if ddp.is_distributed:
            model = DistributedDataParallel(base_model, device_ids=[device.index], output_device=device.index)

        optimizer = AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        if config.train.init_from == "resume" and isinstance(raw_checkpoint, dict):
            optimizer_state = raw_checkpoint.get("optimizer_state")
            if isinstance(optimizer_state, dict):
                optimizer.load_state_dict(optimizer_state)

        splits = load_splits(config.data)
        tokenized_splits = _build_tokenized_split_sources(config, splits, tokenizer, spec)

        train_dataset = RandomDNAGPTWindowDataset(
            sources=tokenized_splits["train"],
            seq_length=config.model.seq_length,
            samples_per_epoch=config.data.train_samples_per_epoch,
            seed=config.train.seed,
            sampling_strategy=config.data.train_sampling_strategy,
            pad_id=tokenizer.pad_id,
        )
        val_dataset = SequentialDNAGPTWindowDataset(
            sources=tokenized_splits["val"],
            seq_length=config.model.seq_length,
            pad_id=tokenizer.pad_id,
        )
        test_dataset = SequentialDNAGPTWindowDataset(
            sources=tokenized_splits["test"],
            seq_length=config.model.seq_length,
            pad_id=tokenizer.pad_id,
        )

        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=True)
            if ddp.is_distributed
            else None
        )
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False)
            if ddp.is_distributed
            else None
        )
        test_sampler = (
            DistributedSampler(test_dataset, num_replicas=ddp.world_size, rank=ddp.rank, shuffle=False)
            if ddp.is_distributed
            else None
        )

        dataloader_pin_memory = bool(config.train.pin_memory and device.type == "cuda")
        dataloader_persistent_workers = bool(config.train.persistent_workers and config.train.num_workers > 0)
        dataloader_common_kwargs: dict[str, object] = {
            "num_workers": config.train.num_workers,
            "pin_memory": dataloader_pin_memory,
            "persistent_workers": dataloader_persistent_workers,
        }
        if config.train.num_workers > 0:
            dataloader_common_kwargs["prefetch_factor"] = config.train.prefetch_factor

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **dataloader_common_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.train.eval_batch_size,
            shuffle=False,
            sampler=val_sampler,
            **dataloader_common_kwargs,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.train.eval_batch_size,
            shuffle=False,
            sampler=test_sampler,
            **dataloader_common_kwargs,
        )

        total_train_steps = config.train.epochs * len(train_loader)
        scheduler = build_lr_scheduler(
            optimizer=optimizer,
            scheduler_type=config.train.lr_scheduler,
            warmup_steps=config.train.lr_warmup_steps,
            total_steps=total_train_steps,
            min_ratio=config.train.lr_min_ratio,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and config.train.dtype == "float16")

        best_val_bpb = float(resume_metadata.get("best_val_bpb", float("inf")))
        global_step = int(resume_metadata.get("step", 0))

        run_summary: dict[str, object] = {
            "device": str(device),
            "gpu_ids": gpu_ids,
            "distributed_data_parallel": ddp.is_distributed,
            "world_size": ddp.world_size,
            "num_workers": config.train.num_workers,
            "prefetch_factor": config.train.prefetch_factor if config.train.num_workers > 0 else None,
            "persistent_workers": dataloader_persistent_workers,
            "pin_memory": dataloader_pin_memory,
            "model_parameters": int(sum(parameter.numel() for parameter in base_model.parameters())),
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
                "requested_init_from": config.train.init_from,
                "loaded_checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "tokenized_train": _dataset_token_summary(tokenized_splits["train"]),
                "tokenized_val": _dataset_token_summary(tokenized_splits["val"]),
                "tokenized_test": _dataset_token_summary(tokenized_splits["test"]),
            },
        }

        if mode in {"train", "all"}:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            log_started = time.time()
            for epoch in range(config.train.epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                for batch in train_loader:
                    global_step += 1
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    base_lengths = batch["base_lengths"].to(device, non_blocking=True)
                    with autocast_context(device, config.train.dtype):
                        logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits.transpose(1, 2).float(),
                        labels,
                        ignore_index=IGNORE_INDEX,
                    )

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
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
                        tokens_per_second = (
                            config.train.batch_size * config.model.seq_length * config.train.log_interval
                        ) / max(time.time() - log_started, 1e-6)
                        if ddp.is_distributed:
                            tokens_per_second *= ddp.world_size
                        print(
                            f"[train] epoch={epoch + 1} step={global_step} "
                            f"loss/token={loss.item():.4f} bits/base={bits_per_base:.4f} "
                            f"tokens/s={tokens_per_second:.1f} lr={optimizer.param_groups[0]['lr']:.6g}"
                        )
                        log_started = time.time()

                    if global_step % config.train.eval_interval == 0:
                        val_metrics = evaluate_dnagpt_loss(
                            model=model,
                            dataloader=val_loader,
                            device=device,
                            dtype_name=config.train.dtype,
                            is_distributed=ddp.is_distributed,
                        )
                        if ddp.is_main_process:
                            print(
                                f"[eval] step={global_step} val_loss/token={val_metrics['loss_nats_per_token']:.4f} "
                                f"val_bits/base={val_metrics['bits_per_base']:.4f}"
                            )
                            if val_metrics["bits_per_base"] < best_val_bpb:
                                best_val_bpb = float(val_metrics["bits_per_base"])
                                save_checkpoint(
                                    output_dir / "best.pt",
                                    unwrap_model(model),
                                    optimizer,
                                    global_step,
                                    best_val_bpb,
                                )
                        model.train()

            if ddp.is_main_process:
                if best_val_bpb == float("inf"):
                    val_metrics = evaluate_dnagpt_loss(
                        model=model,
                        dataloader=val_loader,
                        device=device,
                        dtype_name=config.train.dtype,
                        is_distributed=ddp.is_distributed,
                    )
                    best_val_bpb = float(val_metrics["bits_per_base"])
                    save_checkpoint(
                        output_dir / "best.pt",
                        unwrap_model(model),
                        optimizer,
                        global_step,
                        best_val_bpb,
                    )

                save_checkpoint(
                    output_dir / "last.pt",
                    unwrap_model(model),
                    optimizer,
                    global_step,
                    best_val_bpb,
                )
                run_summary["best_val_bits_per_base"] = best_val_bpb

        if ddp.is_distributed:
            dist.barrier()

        best_checkpoint_path = output_dir / "best.pt"
        if mode in {"train", "all"} and best_checkpoint_path.exists():
            model_state, _, _ = load_dnagpt_checkpoint(best_checkpoint_path, map_location=device)
            unwrap_model(model).load_state_dict(model_state, strict=False)

        if mode in {"eval", "all"}:
            if ddp.is_main_process:
                print("[stage] running final validation...", flush=True)
            val_metrics = evaluate_dnagpt_loss(
                model=model,
                dataloader=val_loader,
                device=device,
                dtype_name=config.train.dtype,
                is_distributed=ddp.is_distributed,
            )
            if ddp.is_main_process:
                print("[stage] running final test evaluation...", flush=True)
            test_metrics = evaluate_dnagpt_loss(
                model=model,
                dataloader=test_loader,
                device=device,
                dtype_name=config.train.dtype,
                is_distributed=ddp.is_distributed,
            )
            if ddp.is_main_process:
                run_summary["validation"] = val_metrics
                run_summary["test"] = test_metrics

        if ddp.is_main_process:
            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            return run_summary

        return {}
    finally:
        if ddp.is_distributed:
            cleanup_distributed()
