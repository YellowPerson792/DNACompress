from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import time
import gc
from typing import Any, Callable, TextIO

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .compression import (
    ArithmeticEncoder,
    baseline_sizes,
    probabilities_to_cumulative_batch,
    resolve_arithmetic_coding_metadata,
)
from .config import ExperimentConfig, save_experiment_config
from .data import (
    RandomWindowDataset,
    SequentialWindowDataset,
    load_splits,
)
from .megabyte_loader import build_model, load_megabyte_checkpoint
from .tokenization import apply_token_merge_to_model_config, tokenize_source_bytes


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class DistributedContext:
    def __init__(self, rank: int, local_rank: int, world_size: int, is_distributed: bool) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = is_distributed

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def resolve_device_and_gpu_ids(device_name: str, gpu_ids: list[int] | None) -> tuple[torch.device, list[int]]:
    requested_device = resolve_device(device_name)
    available_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if requested_device.type != "cuda":
        if gpu_ids:
            raise ValueError("train.gpu_ids is set but train.device is not CUDA. Use train.device=cuda/auto.")
        return requested_device, []

    if available_gpu_count == 0:
        if gpu_ids:
            raise ValueError("train.gpu_ids was provided but CUDA is not available.")
        return torch.device("cpu"), []

    if gpu_ids is not None:
        resolved_gpu_ids = gpu_ids
    elif device_name == "auto":
        resolved_gpu_ids = list(range(available_gpu_count))
    elif requested_device.index is not None:
        resolved_gpu_ids = [int(requested_device.index)]
    else:
        resolved_gpu_ids = list(range(available_gpu_count))

    if not resolved_gpu_ids:
        return torch.device("cpu"), []

    for gpu_id in resolved_gpu_ids:
        if gpu_id < 0 or gpu_id >= available_gpu_count:
            raise ValueError(
                f"Requested GPU id {gpu_id} is out of range for this machine (0..{available_gpu_count - 1})."
            )

    primary_gpu = resolved_gpu_ids[0]
    return torch.device(f"cuda:{primary_gpu}"), resolved_gpu_ids


def setup_distributed_context(device_name: str, gpu_ids: list[int] | None) -> tuple[DistributedContext, torch.device, list[int]]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if not is_distributed:
        device, resolved_gpu_ids = resolve_device_and_gpu_ids(device_name, gpu_ids)
        return DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False), device, resolved_gpu_ids

    requested_device = resolve_device(device_name)
    if requested_device.type != "cuda":
        raise ValueError("DDP requires CUDA devices. Set train.device to auto/cuda/cuda:<id>.")

    if gpu_ids is not None:
        if len(gpu_ids) != world_size:
            raise ValueError(
                f"In DDP mode, number of GPU ids ({len(gpu_ids)}) must match WORLD_SIZE ({world_size})."
            )
        if local_rank < 0 or local_rank >= len(gpu_ids):
            raise ValueError(f"LOCAL_RANK={local_rank} is out of range for gpu_ids={gpu_ids}.")
        resolved_gpu_ids = gpu_ids
        local_gpu_id = gpu_ids[local_rank]
    else:
        if not torch.cuda.is_available():
            raise ValueError("DDP was requested but CUDA is not available.")
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise ValueError(
                f"LOCAL_RANK={local_rank} is out of range for available CUDA devices ({torch.cuda.device_count()})."
            )
        local_gpu_id = local_rank
        resolved_gpu_ids = list(range(world_size))

    device = torch.device(f"cuda:{local_gpu_id}")
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    return (
        DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, is_distributed=True),
        device,
        resolved_gpu_ids,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)):
        return model.module
    return model


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_name)
    if dtype is None:
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype)


def open_training_log_file(output_dir: Path, filename: str = "training_metrics.jsonl") -> tuple[Path, TextIO]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    handle = path.open("a", encoding="utf-8")
    return path, handle


def write_training_log_event(handle: TextIO, event: dict[str, object]) -> None:
    line = json.dumps(event, ensure_ascii=False)
    handle.write(line + "\n")
    handle.flush()


def init_wandb_run(config: ExperimentConfig, output_dir: Path) -> Any | None:
    enabled = bool(config.output.wandb_enabled or config.output.wandb_project)
    if not enabled:
        return None

    if not config.output.wandb_project:
        raise ValueError("W&B is enabled but output.wandb_project is empty.")

    try:
        import wandb
    except ImportError as error:
        raise ImportError("W&B realtime logging requires wandb. Install with: pip install wandb") from error

    run_name = config.output.wandb_name or config.output.run_name or output_dir.name
    tags = config.output.wandb_tags if config.output.wandb_tags else None
    entity = config.output.wandb_entity or None
    group = config.output.wandb_group or None

    return wandb.init(
        project=config.output.wandb_project,
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        mode=config.output.wandb_mode,
        job_type="train",
        dir=str(output_dir),
        config=config.to_dict(),
        reinit=True,
    )


def log_wandb_metrics(wandb_run: Any | None, payload: dict[str, object], step: int | None = None) -> None:
    if wandb_run is None:
        return
    if step is None:
        wandb_run.log(payload)
        return
    wandb_run.log(payload, step=step)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: AdamW, step: int, best_val_bpb: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "best_val_bpb": best_val_bpb,
        },
        path,
    )


def _resolve_initial_checkpoint_path(config: ExperimentConfig, mode: str, output_dir: Path) -> Path | None:
    init_from = config.train.init_from
    explicit = Path(config.model.pretrained_weight_path) if config.model.pretrained_weight_path else None

    if init_from == "scratch":
        if mode in {"eval", "compress"}:
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

    if init_from == "pretrained":
        if explicit is not None:
            return explicit
        raise FileNotFoundError(
            "train.init_from='pretrained' requires model.pretrained_weight_path to be set for Megabyte."
        )

    raise ValueError(f"Unsupported train.init_from value: {init_from}")


def evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype_name: str,
    pad_id: int,
    token_merge_size: int,
    is_distributed: bool,
) -> dict[str, float]:
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            valid_tokens = int((ids != pad_id).sum().item())
            with autocast_context(device, dtype_name):
                output = model(ids, return_loss=True)
            total_nats += float(output.loss.item()) * valid_tokens
            total_tokens += valid_tokens

    if is_distributed:
        reduced = torch.tensor([total_nats, float(total_tokens)], dtype=torch.float64, device=device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total_nats = float(reduced[0].item())
        total_tokens = int(reduced[1].item())

    total_bases = total_tokens * token_merge_size
    average_nats_per_token = total_nats / max(total_tokens, 1)
    average_nats_per_base = total_nats / max(total_bases, 1)
    bits_per_token = average_nats_per_token / math.log(2)
    bits_per_base = average_nats_per_base / math.log(2)
    return {
        "loss_nats_per_token": average_nats_per_token,
        "loss_nats_per_base": average_nats_per_base,
        "bits_per_token": bits_per_token,
        "bits_per_base": bits_per_base,
        "tokens": total_tokens,
        "bases": total_bases,
        "elapsed_seconds": time.time() - start_time,
    }


def evaluate_compression(
    model: torch.nn.Module,
    payload: bytes,
    seq_length: int,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    token_merge_size: int,
    token_merge_alphabet: str,
    arithmetic_frequency_total: int | None,
    arithmetic_target_uniform_mass: float,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, float | int]:
    token_symbols = tokenize_source_bytes(payload, token_merge_size, token_merge_alphabet)
    symbols = token_symbols + [eos_id]

    # Build all causal windows with a single vectorized unfold instead of Python loops.
    symbols_tensor = torch.tensor(symbols, dtype=torch.long)
    padded = torch.full((len(symbols) + seq_length - 1,), pad_id, dtype=torch.long)
    padded[-len(symbols) :] = symbols_tensor
    windows = padded.unfold(0, seq_length, 1)

    encoder = ArithmeticEncoder()
    total_bits = 0.0
    arithmetic_metadata: dict[str, float | int] | None = None

    model.eval()
    total_batches = math.ceil(len(symbols) / batch_size)
    processed_batches = 0
    with torch.no_grad():
        for start in range(0, len(symbols), batch_size):
            batch = windows[start : start + batch_size].to(device, non_blocking=True)
            with autocast_context(device, dtype_name):
                output = model(batch, return_loss=False)
                log_probs = torch.log_softmax(output.lm_logits[:, -1, :], dim=-1).float()

            target_tensor = symbols_tensor[start : start + log_probs.shape[0]].to(device, non_blocking=True)
            picked_log_probs = log_probs.gather(1, target_tensor.unsqueeze(1)).squeeze(1)
            total_bits += float((-picked_log_probs / math.log(2)).sum().item())

            probs_np = log_probs.exp().cpu().numpy()
            targets_np = target_tensor.cpu().numpy()
            if arithmetic_metadata is None:
                arithmetic_metadata = resolve_arithmetic_coding_metadata(
                    vocab_size=int(probs_np.shape[1]),
                    requested_total=arithmetic_frequency_total,
                    target_uniform_mass=arithmetic_target_uniform_mass,
                )
            cumulative_batch = probabilities_to_cumulative_batch(
                probs_np,
                total=int(arithmetic_metadata["arithmetic_frequency_total"]),
            )
            for cumulative, target in zip(cumulative_batch, targets_np):
                encoder.update(cumulative, int(target))

            processed_batches += 1
            if progress_callback is not None:
                progress_callback(processed_batches, total_batches)

    compressed = encoder.finish()
    if arithmetic_metadata is None:
        raise RuntimeError("Failed to resolve arithmetic coding metadata during compression evaluation.")
    baselines = baseline_sizes(payload)
    tokenized_bases = len(token_symbols) * token_merge_size
    bytes_count = len(payload)
    return {
        "sample_bytes": bytes_count,
        "sample_bases": tokenized_bases,
        "sample_symbols_with_eos": len(symbols),
        "theoretical_bits": total_bits,
        "theoretical_bits_per_base": total_bits / max(tokenized_bases, 1),
        "arithmetic_coded_bytes": len(compressed),
        "arithmetic_bits_per_base": (len(compressed) * 8) / max(tokenized_bases, 1),
        **arithmetic_metadata,
        **baselines,
    }


def evaluate_compression_per_source(
    model: torch.nn.Module,
    test_sources: list[bytes],
    source_entries: list[dict[str, object]],
    requested_bytes: int,
    seq_length: int,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    token_merge_size: int,
    token_merge_alphabet: str,
    arithmetic_frequency_total: int | None,
    arithmetic_target_uniform_mass: float,
) -> dict[str, object]:
    per_source: list[dict[str, object]] = []
    total_sample_bytes = 0
    total_sample_bases = 0
    total_theoretical_bits = 0.0
    total_arithmetic_bytes = 0

    def _print_compression_progress(
        source_name: str,
        source_index: int,
        source_total: int,
        batch_done: int,
        batch_total: int,
    ) -> None:
        ratio = 100.0 * batch_done / max(batch_total, 1)
        message = (
            f"\r[compress] source {source_index}/{source_total} ({source_name}) "
            f"batch {batch_done}/{batch_total} ({ratio:5.1f}%)"
        )
        print(message, end="", flush=True)

    source_total = len(test_sources)
    if source_total == 0:
        print("[compress] no test sources found.")

    for source_index, (source, entry) in enumerate(zip(test_sources, source_entries), start=1):
        species_name = str(entry["species"])
        source_name = str(entry.get("source_name", species_name))
        payload = source[:requested_bytes] if len(source) >= requested_bytes else source
        metrics = evaluate_compression(
            model=model,
            payload=payload,
            seq_length=seq_length,
            pad_id=pad_id,
            eos_id=eos_id,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
            token_merge_size=token_merge_size,
            token_merge_alphabet=token_merge_alphabet,
            arithmetic_frequency_total=arithmetic_frequency_total,
            arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
            progress_callback=lambda batch_done, batch_total, si=source_index, sn=source_name: _print_compression_progress(
                sn,
                si,
                source_total,
                batch_done,
                batch_total,
            ),
        )
        print()
        per_source.append({"species": species_name, "source_name": source_name, **metrics})
        total_sample_bytes += int(metrics["sample_bytes"])
        total_sample_bases += int(metrics["sample_bases"])
        total_theoretical_bits += float(metrics["theoretical_bits"])
        total_arithmetic_bytes += int(metrics["arithmetic_coded_bytes"])

    if source_total > 0:
        print("[compress] completed.")

    aggregate = {
        "source_count": len(per_source),
        "total_sample_bytes": total_sample_bytes,
        "total_sample_bases": total_sample_bases,
        "total_theoretical_bits": total_theoretical_bits,
        "total_theoretical_bits_per_base": total_theoretical_bits / max(total_sample_bases, 1),
        "total_arithmetic_coded_bytes": total_arithmetic_bytes,
        "total_arithmetic_bits_per_base": (total_arithmetic_bytes * 8) / max(total_sample_bases, 1),
    }
    for key in (
        "arithmetic_frequency_total",
        "arithmetic_vocab_size",
        "arithmetic_target_uniform_mass",
        "arithmetic_effective_uniform_mass",
    ):
        if per_source and all(row.get(key) == per_source[0].get(key) for row in per_source):
            aggregate[key] = per_source[0].get(key)
    return {
        "aggregate": aggregate,
        "per_source": per_source,
    }


def build_lr_scheduler(
    optimizer: AdamW,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    min_ratio: float,
) -> LambdaLR | None:
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "none":
        return None

    warmup_steps = max(0, warmup_steps)
    total_steps = max(1, total_steps)
    min_ratio = max(0.0, min(1.0, min_ratio))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))

        if scheduler_type == "linear":
            return 1.0 - (1.0 - min_ratio) * progress

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_experiment(config: ExperimentConfig, mode: str = "all") -> dict[str, object]:
    seed_everything(config.train.seed)
    ddp, device, gpu_ids = setup_distributed_context(config.train.device, config.train.gpu_ids)
    train_log_handle: TextIO | None = None
    wandb_run = None
    try:
        apply_token_merge_to_model_config(config.model, config.data)
        output_dir = Path(config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if ddp.is_main_process:
            save_experiment_config(config, output_dir / "resolved_config.json")
            training_log_path, train_log_handle = open_training_log_file(output_dir)
            wandb_run = init_wandb_run(config, output_dir)

        checkpoint_path = _resolve_initial_checkpoint_path(config, mode, output_dir)
        resume_metadata: dict[str, object] = {}
        raw_checkpoint: dict[str, object] | None = None
        optimizer_state_restored = False
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []

        if ddp.is_main_process:
            print("[stage] loading data splits...", flush=True)
        split_started = time.time()
        splits = load_splits(config.data, seq_length=config.model.seq_length)
        if ddp.is_main_process:
            split_entries = splits.summary["species"]
            total_train_bytes = int(sum(int(item["train_bytes"]) for item in split_entries))
            total_val_bytes = int(sum(int(item["val_bytes"]) for item in split_entries))
            total_test_bytes = int(sum(int(item["test_bytes"]) for item in split_entries))
            clean_cache_summary = splits.summary.get("clean_cache", {})
            print(
                "[stage] data splits loaded: "
                f"sources={len(split_entries)} "
                f"train_bytes={total_train_bytes} "
                f"val_bytes={total_val_bytes} "
                f"test_bytes={total_test_bytes} "
                f"elapsed={time.time() - split_started:.1f}s",
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

        if ddp.is_main_process:
            print("[stage] building training dataset...", flush=True)
        train_dataset = RandomWindowDataset(
            sources=splits.train_sources,
            seq_length=config.model.seq_length,
            samples_per_epoch=config.data.train_samples_per_epoch,
            seed=config.train.seed,
            sampling_strategy=config.data.train_sampling_strategy,
            token_merge_size=config.data.token_merge_size,
            token_merge_alphabet=config.data.token_merge_alphabet,
        )
        splits.train_sources = []
        gc.collect()
        if ddp.is_main_process:
            print(
                f"[stage] training dataset ready: tokenized_sources={len(train_dataset.sources)} "
                f"samples_per_epoch={len(train_dataset)}",
                flush=True,
            )

        if ddp.is_main_process:
            print("[stage] building validation dataset...", flush=True)
        val_dataset = SequentialWindowDataset(
            sources=splits.val_sources,
            seq_length=config.model.seq_length,
            pad_id=config.model.pad_id,
            token_merge_size=config.data.token_merge_size,
            token_merge_alphabet=config.data.token_merge_alphabet,
        )
        splits.val_sources = []
        gc.collect()
        if ddp.is_main_process:
            print(
                f"[stage] validation dataset ready: tokenized_sources={len(val_dataset.sources)} "
                f"windows={len(val_dataset)}",
                flush=True,
            )

        if ddp.is_main_process:
            print("[stage] building test dataset...", flush=True)
        test_dataset = SequentialWindowDataset(
            sources=splits.test_sources,
            seq_length=config.model.seq_length,
            pad_id=config.model.pad_id,
            token_merge_size=config.data.token_merge_size,
            token_merge_alphabet=config.data.token_merge_alphabet,
        )
        compression_test_sources = splits.test_sources if mode in {"eval", "all", "compress"} else []
        splits.test_sources = []
        gc.collect()
        if ddp.is_main_process:
            print(
                f"[stage] test dataset ready: tokenized_sources={len(test_dataset.sources)} "
                f"windows={len(test_dataset)}",
                flush=True,
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

        base_model = build_model(config.model)
        if checkpoint_path is not None:
            model_state, resume_metadata, raw_checkpoint = load_megabyte_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )
            load_result = base_model.load_state_dict(
                model_state,
                strict=config.train.init_from == "resume",
            )
            if config.train.init_from == "pretrained":
                missing_keys = list(load_result.missing_keys)
                unexpected_keys = list(load_result.unexpected_keys)

        base_model = base_model.to(device)
        model: torch.nn.Module = base_model
        if ddp.is_distributed:
            model = DistributedDataParallel(base_model, device_ids=[device.index], output_device=device.index)
        optimizer = AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
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

        if config.train.init_from == "resume" and isinstance(raw_checkpoint, dict):
            optimizer_state = raw_checkpoint.get("optimizer_state")
            if isinstance(optimizer_state, dict):
                optimizer.load_state_dict(optimizer_state)
                optimizer_state_restored = True

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
            "megabyte": {
                "implementation": config.model.implementation,
                "requested_init_from": config.train.init_from,
                "loaded_checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "optimizer_state_restored": optimizer_state_restored,
                "pretrained_load_missing_keys": missing_keys,
                "pretrained_load_unexpected_keys": unexpected_keys,
            },
        }
        if ddp.is_main_process and train_log_handle is not None:
            run_summary["training_log_jsonl"] = str(training_log_path)

        # Only resume mode should inherit optimizer progress/step counters.
        # Pretrained initialization should load model weights but restart training stats.
        if config.train.init_from == "resume":
            best_val_bpb = float(resume_metadata.get("best_val_bpb", float("inf")))
            global_step = int(resume_metadata.get("step", 0))
        else:
            best_val_bpb = float("inf")
            global_step = 0

        if ddp.is_main_process and checkpoint_path is not None:
            print(
                f"[startup] loaded checkpoint={checkpoint_path} init_from={config.train.init_from} "
                f"optimizer_restored={optimizer_state_restored}",
                flush=True,
            )
            if config.train.init_from == "pretrained" and (missing_keys or unexpected_keys):
                print(
                    f"[startup] pretrained weight load strict=False "
                    f"missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}",
                    flush=True,
                )

        if mode in {"train", "all"}:
            if ddp.is_main_process:
                print("[stage] starting training...", flush=True)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            started = time.time()
            for epoch in range(config.train.epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                for batch in train_loader:
                    global_step += 1
                    ids = batch["input_ids"].to(device, non_blocking=True)
                    with autocast_context(device, config.train.dtype):
                        output = model(ids, return_loss=True)
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
                        tokens_per_second = (
                            config.train.batch_size * config.model.seq_length * config.train.log_interval
                        ) / max(time.time() - started, 1e-6)
                        if ddp.is_distributed:
                            tokens_per_second *= ddp.world_size
                        bases_per_second = tokens_per_second * config.data.token_merge_size
                        bytes_per_second = bases_per_second
                        bits_per_base = (loss.item() / math.log(2)) / config.data.token_merge_size
                        if train_log_handle is not None:
                            write_training_log_event(
                                train_log_handle,
                                {
                                    "event": "train",
                                    "step": global_step,
                                    "epoch": epoch + 1,
                                    "loss_nats_per_token": float(loss.item()),
                                    "bits_per_base": float(bits_per_base),
                                    "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                                    "tokens_per_second": float(tokens_per_second),
                                },
                            )
                        log_wandb_metrics(
                            wandb_run,
                            {
                                "epoch": epoch + 1,
                                "train/loss": float(loss.item()),
                                "train/bpb": float(bits_per_base),
                                "train/grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                                "train/lr": float(optimizer.param_groups[0]["lr"]),
                                "train/tokens_per_second": float(tokens_per_second),
                            },
                            step=global_step,
                        )
                        print(
                            f"[train] epoch={epoch + 1} step={global_step} "
                            f"loss/token={loss.item():.4f} bits/base={bits_per_base:.4f} "
                            f"grad_norm={float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm):.4f} "
                            f"bytes/s={bytes_per_second:.1f} lr={optimizer.param_groups[0]['lr']:.6g}",
                            flush=True,
                        )
                        started = time.time()

                    if global_step % config.train.eval_interval == 0:
                        val_metrics = evaluate_loss(
                            model,
                            val_loader,
                            device,
                            config.train.dtype,
                            config.model.pad_id,
                            config.data.token_merge_size,
                            is_distributed=ddp.is_distributed,
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
                                    "val/loss": float(val_metrics["loss_nats_per_token"]),
                                    "val/bpb": float(val_metrics["bits_per_base"]),
                                },
                                step=global_step,
                            )
                            print(
                                f"[eval] step={global_step} val_loss/token={val_metrics['loss_nats_per_token']:.4f} "
                                f"val_bits/base={val_metrics['bits_per_base']:.4f}",
                                flush=True,
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

            if best_val_bpb == float("inf"):
                val_metrics = evaluate_loss(
                    model,
                    val_loader,
                    device,
                    config.train.dtype,
                    config.model.pad_id,
                    config.data.token_merge_size,
                    is_distributed=ddp.is_distributed,
                )
                if ddp.is_main_process:
                    best_val_bpb = float(val_metrics["bits_per_base"])
                    save_checkpoint(
                        output_dir / "best.pt",
                        unwrap_model(model),
                        optimizer,
                        global_step,
                        best_val_bpb,
                    )

            if ddp.is_main_process:
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
            model_state, _, _ = load_megabyte_checkpoint(best_checkpoint_path, map_location=device)
            unwrap_model(model).load_state_dict(model_state)

        if mode in {"eval", "all", "compress"}:
            if ddp.is_main_process:
                print("[stage] running final validation...", flush=True)
            val_metrics = evaluate_loss(
                model,
                val_loader,
                device,
                config.train.dtype,
                config.model.pad_id,
                config.data.token_merge_size,
                is_distributed=ddp.is_distributed,
            )
            if ddp.is_main_process:
                print("[stage] running final test evaluation...", flush=True)
            test_metrics = evaluate_loss(
                model,
                test_loader,
                device,
                config.train.dtype,
                config.model.pad_id,
                config.data.token_merge_size,
                is_distributed=ddp.is_distributed,
            )

            if ddp.is_main_process:
                if train_log_handle is not None:
                    write_training_log_event(
                        train_log_handle,
                        {
                            "event": "eval",
                            "split": "val",
                            "step": global_step,
                            "epoch": config.train.epochs,
                            "loss_nats_per_token": float(val_metrics["loss_nats_per_token"]),
                            "bits_per_base": float(val_metrics["bits_per_base"]),
                            "tokens": int(val_metrics["tokens"]),
                            "bases": int(val_metrics["bases"]),
                            "is_final": True,
                        },
                    )
                    write_training_log_event(
                        train_log_handle,
                        {
                            "event": "eval",
                            "split": "test",
                            "step": global_step,
                            "epoch": config.train.epochs,
                            "loss_nats_per_token": float(test_metrics["loss_nats_per_token"]),
                            "bits_per_base": float(test_metrics["bits_per_base"]),
                            "tokens": int(test_metrics["tokens"]),
                            "bases": int(test_metrics["bases"]),
                            "is_final": True,
                        },
                    )
                log_wandb_metrics(
                    wandb_run,
                    {
                        "epoch": config.train.epochs,
                        "eval/final_loss": float(val_metrics["loss_nats_per_token"]),
                        "eval/final_bpb": float(val_metrics["bits_per_base"]),
                        "val/final_loss": float(val_metrics["loss_nats_per_token"]),
                        "val/final_bpb": float(val_metrics["bits_per_base"]),
                        "test/loss": float(test_metrics["loss_nats_per_token"]),
                        "test/bpb": float(test_metrics["bits_per_base"]),
                    },
                    step=global_step,
                )
                source_entries = [dict(item) for item in splits.summary["species"]]
                # Compression runs only on rank 0. Use the underlying module directly so
                # rank-local forward passes do not depend on other DDP ranks staying active.
                print("[stage] starting compression on rank 0...", flush=True)
                compression_model = unwrap_model(model)
                compression_metrics = evaluate_compression_per_source(
                    model=compression_model,
                    test_sources=compression_test_sources,
                    source_entries=source_entries,
                    requested_bytes=config.data.compression_sample_bytes,
                    seq_length=config.model.seq_length,
                    pad_id=config.model.pad_id,
                    eos_id=config.model.eos_id,
                    device=device,
                    dtype_name=config.train.dtype,
                    batch_size=config.train.eval_batch_size,
                    token_merge_size=config.data.token_merge_size,
                    token_merge_alphabet=config.data.token_merge_alphabet,
                    arithmetic_frequency_total=config.arithmetic.frequency_total,
                    arithmetic_target_uniform_mass=config.arithmetic.target_uniform_mass,
                )

                run_summary["validation"] = val_metrics
                run_summary["test"] = test_metrics
                run_summary["compression"] = compression_metrics

            if ddp.is_distributed:
                # Keep non-main ranks alive until rank 0 finishes standalone compression.
                dist.barrier()

        if ddp.is_main_process:
            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(run_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if wandb_run is not None:
                if "best_val_bits_per_base" in run_summary:
                    wandb_run.summary["best_val/bpb"] = run_summary["best_val_bits_per_base"]
                if "model_parameters" in run_summary:
                    wandb_run.summary["model_parameters"] = run_summary["model_parameters"]
                wandb_run.summary["output_dir"] = str(output_dir)
            return run_summary

        return {}
    finally:
        if train_log_handle is not None:
            train_log_handle.close()
        if wandb_run is not None:
            wandb_run.finish()
        if ddp.is_distributed:
            cleanup_distributed()
